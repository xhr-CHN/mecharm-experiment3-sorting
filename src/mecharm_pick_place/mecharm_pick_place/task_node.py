"""ROS 2 node that runs the platform-independent pick-and-place sequence."""

from __future__ import annotations

import json
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

from .result_types import ErrorCode, TaskEvent, TaskState
from .state_machine import PickPlaceStateMachine


_POSE_STATES = {
    TaskState.ABOVE_PICK: ("pick", True),
    TaskState.DESCEND_PICK: ("pick", False),
    TaskState.LIFT_PICK: ("pick", True),
    TaskState.ABOVE_PLACE: ("place", True),
    TaskState.DESCEND_PLACE: ("place", False),
    TaskState.RETREAT: ("place", True),
}


class PickPlaceTaskNode(Node):
    def __init__(self) -> None:
        super().__init__("pick_place_task_node")
        self._declare_parameters()
        self.machine = PickPlaceStateMachine()
        self.attempt_id = 0
        self.batch_active = False
        self.waiting_for_result = False
        self.command_started = 0.0
        self.next_attempt_at = 0.0

        self.pose_pub = self.create_publisher(PoseStamped, "/mecharm/target_pose", 10)
        self.joint_pub = self.create_publisher(JointState, "/mecharm/joint_target", 10)
        self.gripper_pub = self.create_publisher(Float64, "/mecharm/gripper_command", 10)
        self.status_pub = self.create_publisher(String, "/mecharm/task_status", 10)
        self.result_pub = self.create_publisher(String, "/mecharm/task_result", 10)
        self.create_subscription(String, "/mecharm/motion_result", self._on_motion_result, 10)
        self.create_service(Trigger, "/mecharm/start_task", self._start_task)
        self.create_service(Trigger, "/mecharm/stop_task", self._stop_task)
        self.create_timer(0.05, self._tick)

    def _declare_parameters(self) -> None:
        self.declare_parameter("frame_id", "base")
        self.declare_parameter(
            "home_joint_names",
            [
                "joint1_to_base",
                "joint2_to_joint1",
                "joint3_to_joint2",
                "joint4_to_joint3",
                "joint5_to_joint4",
                "joint6_to_joint5",
            ],
        )
        self.declare_parameter("home_joint_positions", [0.0] * 6)
        self.declare_parameter("pick_position", [0.18, 0.08, 0.05])
        self.declare_parameter("place_position", [0.18, -0.08, 0.05])
        self.declare_parameter("safe_height", 0.14)
        self.declare_parameter("tool_orientation_xyzw", [0.0, 1.0, 0.0, 0.0])
        self.declare_parameter("gripper_open", 1.0)
        self.declare_parameter("gripper_closed", 0.0)
        self.declare_parameter("motion_timeout_sec", 8.0)
        self.declare_parameter("gripper_timeout_sec", 3.0)
        self.declare_parameter("inter_attempt_delay_sec", 2.0)
        self.declare_parameter("attempts_per_batch", 5)
        self.declare_parameter("minimum_successes", 4)

    def _start_task(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.batch_active or self.machine.active:
            response.success = False
            response.message = "a pick-and-place batch is already active"
            return response
        self.attempt_id = 0
        self.batch_active = True
        self._begin_attempt()
        response.success = True
        response.message = "started five-attempt pick-and-place batch"
        return response

    def _stop_task(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self.batch_active and not self.machine.active:
            response.success = False
            response.message = "no active task"
            return response
        self.batch_active = False
        self.waiting_for_result = False
        self.machine.fail(ErrorCode.STOP_REQUESTED, "operator requested stop")
        self._finish_attempt(False)
        response.success = True
        response.message = "task stopped"
        return response

    def _begin_attempt(self) -> None:
        self.attempt_id += 1
        self.machine.start()
        self.waiting_for_result = False
        self._publish_status(True, "attempt started")

    def _tick(self) -> None:
        now = time.monotonic()
        if self.next_attempt_at and now >= self.next_attempt_at:
            self.next_attempt_at = 0.0
            self._begin_attempt()
            return
        if not self.machine.active:
            return
        if self.waiting_for_result:
            timeout = self._timeout_for_state(self.machine.state)
            if now - self.command_started > timeout:
                self.machine.fail(ErrorCode.TIMEOUT, f"{self.machine.state.value} timed out")
                self._finish_attempt(False)
            return
        self._dispatch_current_state()

    def _dispatch_current_state(self) -> None:
        state = self.machine.state
        self._publish_status(True, "state entered")
        if state in {TaskState.HOME, TaskState.RETURN_HOME}:
            self._publish_home()
            self._wait_for_result()
        elif state in _POSE_STATES:
            location, use_safe_height = _POSE_STATES[state]
            self._publish_pose(location, use_safe_height)
            self._wait_for_result()
        elif state is TaskState.CLOSE_GRIPPER:
            self.gripper_pub.publish(Float64(data=float(self.get_parameter("gripper_closed").value)))
            self._wait_for_result()
        elif state is TaskState.OPEN_GRIPPER:
            self.gripper_pub.publish(Float64(data=float(self.get_parameter("gripper_open").value)))
            self._wait_for_result()
        elif state in {TaskState.ATTACH_OBJECT, TaskState.DETACH_OBJECT}:
            self.machine.advance()
        else:
            self.machine.fail(ErrorCode.MOTION_FAILED, f"unsupported state: {state.value}")
            self._finish_attempt(False)

    def _publish_home(self) -> None:
        names = list(self.get_parameter("home_joint_names").value)
        positions = [float(value) for value in self.get_parameter("home_joint_positions").value]
        if not names or len(names) != len(positions):
            self.machine.fail(ErrorCode.JOINT_LIMIT, "home joint configuration is invalid")
            self._finish_attempt(False)
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        self.joint_pub.publish(msg)

    def _publish_pose(self, location: str, use_safe_height: bool) -> None:
        values = list(self.get_parameter(f"{location}_position").value)
        orientation = list(self.get_parameter("tool_orientation_xyzw").value)
        if len(values) != 3 or len(orientation) != 4:
            self.machine.fail(ErrorCode.INVALID_RESULT, "pose configuration has invalid dimensions")
            self._finish_attempt(False)
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter("frame_id").value)
        msg.pose.position.x = float(values[0])
        msg.pose.position.y = float(values[1])
        msg.pose.position.z = float(self.get_parameter("safe_height").value if use_safe_height else values[2])
        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])
        self.pose_pub.publish(msg)

    def _wait_for_result(self) -> None:
        if self.machine.state is TaskState.FAILED:
            return
        self.waiting_for_result = True
        self.command_started = time.monotonic()

    def _on_motion_result(self, msg: String) -> None:
        if not self.waiting_for_result or not self.machine.active:
            return
        try:
            data = json.loads(msg.data)
            success = bool(data["success"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.machine.fail(ErrorCode.INVALID_RESULT, f"invalid bridge result: {exc}")
            self.waiting_for_result = False
            self._finish_attempt(False)
            return

        self.waiting_for_result = False
        if not success:
            raw_code = str(data.get("error_code", ErrorCode.MOTION_FAILED.value))
            try:
                code = ErrorCode(raw_code)
            except ValueError:
                code = ErrorCode.MOTION_FAILED
            self.machine.fail(code, str(data.get("message", "bridge rejected command")))
            self._finish_attempt(False)
            return

        next_state = self.machine.advance()
        if next_state is TaskState.SUCCESS:
            self._finish_attempt(True)

    def _finish_attempt(self, success: bool) -> None:
        event = TaskEvent(
            attempt_id=self.attempt_id,
            state=self.machine.state.value,
            success=success,
            error_code=self.machine.error_code.value,
            message=self.machine.error_message or ("attempt completed" if success else "attempt failed"),
        )
        self.result_pub.publish(String(data=event.to_json()))
        attempts = int(self.get_parameter("attempts_per_batch").value)
        if self.batch_active and self.attempt_id < attempts:
            self.machine.reset()
            self.next_attempt_at = time.monotonic() + float(
                self.get_parameter("inter_attempt_delay_sec").value
            )
        else:
            self.batch_active = False

    def _publish_status(self, success: bool, message: str) -> None:
        event = TaskEvent(
            attempt_id=self.attempt_id,
            state=self.machine.state.value,
            success=success,
            error_code=self.machine.error_code.value,
            message=message,
        )
        self.status_pub.publish(String(data=event.to_json()))

    def _timeout_for_state(self, state: TaskState) -> float:
        parameter = "gripper_timeout_sec" if state in {
            TaskState.CLOSE_GRIPPER,
            TaskState.OPEN_GRIPPER,
        } else "motion_timeout_sec"
        return float(self.get_parameter(parameter).value)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickPlaceTaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
