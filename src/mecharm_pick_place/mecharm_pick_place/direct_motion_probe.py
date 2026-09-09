"""Collision-planner-free smooth motion probe for transport validation."""

from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState


JOINT_NAMES = (
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "gripper_controller",
)


class DirectMotionProbe(Node):
    def __init__(self, node_name="direct_motion_probe"):
        super().__init__(node_name)
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.publisher = self.create_publisher(
            JointState, "/mecharm/joint_target", 10
        )
        callback_group = ReentrantCallbackGroup()
        self.subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self._state_callback,
            state_qos,
            callback_group=callback_group,
        )
        self.positions = None
        self.state_sequence = 0

    def _state_callback(self, message):
        values = dict(zip(message.name, message.position))
        if all(name in values for name in JOINT_NAMES):
            positions = tuple(float(values[name]) for name in JOINT_NAMES)
            if all(math.isfinite(value) for value in positions):
                self.positions = positions
                self.state_sequence += 1

    def publish_positions(self, positions):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = list(positions)
        self.publisher.publish(message)


def _smoothstep(value):
    value = min(1.0, max(0.0, value))
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _execute_segment(node, start, target, duration=3.0, rate_hz=120.0):
    if duration <= 0.0 or rate_hz <= 0.0:
        raise ValueError("duration and rate_hz must be positive")
    period = 1.0 / rate_hz
    elapsed = 0.0
    external_executor_spins = bool(
        getattr(node, "external_executor_spins", False)
    )
    # Use elapsed time without coupling motion speed to feedback frequency.
    # Cap a scheduling gap to two periods to avoid catch-up target jumps.
    if not external_executor_spins:
        rclpy.spin_once(node, timeout_sec=0.0)
    sequence = node.state_sequence
    node.publish_positions(start)
    last_feedback = time.monotonic()
    previous_tick = last_feedback
    while rclpy.ok():
        tick = time.monotonic()
        if not external_executor_spins:
            rclpy.spin_once(node, timeout_sec=0.0)
        if node.state_sequence != sequence:
            sequence = node.state_sequence
            last_feedback = tick
        feedback_age = tick - last_feedback
        if feedback_age > 5.0:
            raise RuntimeError("DIRECT_MOTION_FEEDBACK_TIMEOUT: holding last target")
        if feedback_age <= 0.25:
            elapsed = min(duration, elapsed + min(tick - previous_tick, 2.0 * period))
        previous_tick = tick
        ratio = _smoothstep(elapsed / duration)
        command = tuple(a + ratio * (b - a) for a, b in zip(start, target))
        node.publish_positions(command)
        if elapsed >= duration:
            break
        time.sleep(max(0.0, period - (time.monotonic() - tick)))


def main(args=None):
    rclpy.init(args=args)
    node = DirectMotionProbe()
    try:
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and node.positions is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.positions is None:
            node.get_logger().error("MOTION_PROBE_NO_JOINT_STATE")
            return
        start = node.positions
        target = list(start)
        direction = 1.0 if start[0] <= 2.4 else -1.0
        target[0] = max(-2.7, min(2.7, start[0] + direction * 0.25))
        if not all(math.isfinite(value) for value in start):
            node.get_logger().error("MOTION_PROBE_INVALID_JOINT_STATE")
            return
        node.get_logger().info("MOTION_PROBE_OUTBOUND")
        _execute_segment(node, start, tuple(target))
        node.get_logger().info("MOTION_PROBE_RETURN")
        _execute_segment(node, tuple(target), start)
        for _ in range(10):
            node.publish_positions(start)
            time.sleep(1.0 / 120.0)
        node.get_logger().info("MOTION_PROBE_SUCCESS")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
