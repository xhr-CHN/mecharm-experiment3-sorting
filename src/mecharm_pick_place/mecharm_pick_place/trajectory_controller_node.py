"""FollowJointTrajectory adapter that streams named targets to Isaac Sim."""

from __future__ import annotations

import time
from threading import Lock

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectoryPoint

from .trajectory import RawPoint, execution_outcome, normalize_trajectory, sample_trajectory


ARM_JOINTS = (
    "joint1_to_base", "joint2_to_joint1", "joint3_to_joint2",
    "joint4_to_joint3", "joint5_to_joint4", "joint6_to_joint5",
)
DEFAULT_LIMITS = {
    "joint1_to_base": (-2.792527, 2.792527),
    "joint2_to_joint1": (-1.3089, 2.0943),
    "joint3_to_joint2": (-3.0543, 1.1344),
    "joint4_to_joint3": (-2.7052, 2.7052),
    "joint5_to_joint4": (-2.0071, 2.0071),
    "joint6_to_joint5": (-3.14, 3.14),
}


class IsaacTrajectoryController(Node):
    def __init__(self):
        super().__init__("isaac_trajectory_controller")
        self.declare_parameter("publish_rate_hz", 120.0)
        self.declare_parameter("joint_tolerance", 0.02)
        self.declare_parameter("feedback_stale_timeout_sec", 0.5)
        self.declare_parameter("goal_time_tolerance_sec", 1.0)
        self.declare_parameter("gripper_open_position", 0.15)
        self.declare_parameter("gripper_closed_position", -0.75)
        self.rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.tolerance = float(self.get_parameter("joint_tolerance").value)
        self.stale_timeout = float(self.get_parameter("feedback_stale_timeout_sec").value)
        self.goal_time_tolerance = float(self.get_parameter("goal_time_tolerance_sec").value)
        self.gripper_open = float(self.get_parameter("gripper_open_position").value)
        self.gripper_closed = float(self.get_parameter("gripper_closed_position").value)
        self._lock = Lock()
        self._measured = {}
        self._feedback_stamp = None
        self._last_arm_target = tuple(0.0 for _ in ARM_JOINTS)
        self._gripper_target = self.gripper_open
        group = ReentrantCallbackGroup()
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._command_pub = self.create_publisher(JointState, "/mecharm/joint_target", 20)
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            state_qos,
            callback_group=group,
        )
        self.create_subscription(Float64, "/mecharm/gripper_command", self._on_gripper, 10, callback_group=group)
        self._action = ActionServer(
            self,
            FollowJointTrajectory,
            "/mecharm_controller/follow_joint_trajectory",
            execute_callback=self._execute,
            goal_callback=self._accept_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=group,
        )
        self.get_logger().info("trajectory action ready: /mecharm_controller/follow_joint_trajectory")

    @staticmethod
    def _raw_trajectory(request):
        points = tuple(
            RawPoint(
                tuple(point.positions),
                point.time_from_start.sec + point.time_from_start.nanosec * 1e-9,
            )
            for point in request.trajectory.points
        )
        return normalize_trajectory(request.trajectory.joint_names, points, ARM_JOINTS, DEFAULT_LIMITS)

    def _accept_goal(self, request):
        try:
            self._raw_trajectory(request)
        except ValueError as exc:
            self.get_logger().error(f"rejecting trajectory: {exc}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_joint_state(self, msg):
        with self._lock:
            self._measured.update({name: float(value) for name, value in zip(msg.name, msg.position)})
            self._feedback_stamp = self.get_clock().now()

    def _on_gripper(self, msg):
        opening = max(0.0, min(1.0, float(msg.data)))
        with self._lock:
            self._gripper_target = self.gripper_closed + opening * (
                self.gripper_open - self.gripper_closed
            )
            arm = self._last_arm_target
        self._publish_target(arm)

    def _publish_target(self, arm_positions):
        with self._lock:
            self._last_arm_target = tuple(arm_positions)
            gripper = self._gripper_target
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [*ARM_JOINTS, "gripper_controller"]
        msg.position = [*self._last_arm_target, gripper]
        self._command_pub.publish(msg)

    def _snapshot(self):
        now = self.get_clock().now()
        with self._lock:
            measured = tuple(self._measured.get(name, float("nan")) for name in ARM_JOINTS)
            age = float("inf") if self._feedback_stamp is None else (now - self._feedback_stamp).nanoseconds * 1e-9
        return measured, age

    def _hold(self):
        measured, _ = self._snapshot()
        if all(value == value for value in measured):
            self._publish_target(measured)

    def _execute(self, goal_handle):
        trajectory = self._raw_trajectory(goal_handle.request)
        start = self.get_clock().now()
        duration = trajectory.points[-1].time_from_start
        result = FollowJointTrajectory.Result()
        while rclpy.ok():
            elapsed = (self.get_clock().now() - start).nanoseconds * 1e-9
            desired = sample_trajectory(trajectory, elapsed)
            self._publish_target(desired)
            measured, feedback_age = self._snapshot()
            finite_feedback = all(value == value for value in measured)
            final_error = max(abs(a - b) for a, b in zip(measured, trajectory.points[-1].positions)) if finite_feedback else float("inf")

            feedback = FollowJointTrajectory.Feedback()
            feedback.joint_names = list(ARM_JOINTS)
            feedback.desired = JointTrajectoryPoint(positions=list(desired))
            feedback.actual = JointTrajectoryPoint(positions=list(measured))
            feedback.error = JointTrajectoryPoint(
                positions=[a - d for a, d in zip(measured, desired)]
            )
            goal_handle.publish_feedback(feedback)

            if goal_handle.is_cancel_requested:
                self._hold()
                goal_handle.canceled()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                result.error_string = "goal canceled; holding measured pose"
                return result

            outcome = execution_outcome(
                final_error, feedback_age, elapsed, duration, self.tolerance,
                self.stale_timeout, self.goal_time_tolerance,
            )
            if outcome == "succeeded":
                goal_handle.succeed()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return result
            if outcome != "running":
                self._hold()
                goal_handle.abort()
                result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                result.error_string = outcome
                return result
            time.sleep(1.0 / self.rate_hz)

        self._hold()
        goal_handle.abort()
        result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
        result.error_string = "ROS shutdown during execution"
        return result


def main(args=None):
    rclpy.init(args=args)
    node = IsaacTrajectoryController()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
