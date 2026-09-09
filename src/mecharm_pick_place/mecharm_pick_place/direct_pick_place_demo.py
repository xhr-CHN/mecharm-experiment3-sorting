"""Run the calibrated pick-and-place sequence without a collision planner."""

from __future__ import annotations

import math
import time

import rclpy

from .direct_motion_probe import DirectMotionProbe, _execute_segment


HOME = (0.0, 0.0, 0.0, -47.0, 0.0, 0.0)
PICK_SAFE = (23.093, -0.417, 18.790, -47.647, -2.617, 0.0)
PICK_GRASP = (23.363, 14.921, 20.671, -47.479, -1.886, 0.0)
PLACE_SAFE = (-19.315, 0.478, 13.209, -47.445, 12.505, 0.0)
PLACE_GRASP = (-19.656, 15.759, 15.709, -47.310, 11.583, 0.0)

MAX_ARM_SPEED_RAD_S = math.radians(20.0)
VERTICAL_ARM_SPEED_RAD_S = math.radians(3.0)
MAX_GRIPPER_SPEED_RAD_S = 0.10
MIN_DURATION_S = 1.0
SETTLE_S = 0.35
GRIPPER_OPEN = 0.15
GRIPPER_CLOSED = -0.75
VERTICAL_STAGE_LABELS = frozenset(
    {"PICK_GRASP", "LIFT", "PLACE_GRASP", "RETREAT"}
)


def _duration(start, target, max_speed):
    largest = max(abs(a - b) for a, b in zip(start, target))
    return max(MIN_DURATION_S, largest / max_speed)


def _move(node, label, command, target):
    # Quintic smoothstep has a peak derivative of 1.875. Account for it so
    # MAX_ARM_SPEED_RAD_S limits peak speed, rather than average speed.
    speed = (
        VERTICAL_ARM_SPEED_RAD_S
        if label in VERTICAL_STAGE_LABELS
        else MAX_ARM_SPEED_RAD_S
    )
    duration = _duration(command, target, speed / 1.875)
    node.get_logger().info(
        f"DIRECT_STAGE {label} duration={duration:.2f}s "
        f"speed={math.degrees(speed):.1f}deg/s"
    )
    _execute_segment(node, command, target, duration=duration)
    time.sleep(SETTLE_S)
    return target


def _move_arm(node, label, command, target_degrees):
    target = tuple(math.radians(value) for value in target_degrees) + (command[6],)
    return _move(node, label, command, target)


def _move_gripper(node, label, command, target_position):
    target = command[:6] + (target_position,)
    duration = _duration((command[6],), (target_position,), MAX_GRIPPER_SPEED_RAD_S)
    node.get_logger().info(f"DIRECT_STAGE {label} duration={duration:.2f}s")
    _execute_segment(node, command, target, duration=duration)
    time.sleep(SETTLE_S)
    return target


def main(args=None):
    rclpy.init(args=args)
    node = DirectMotionProbe(node_name="direct_pick_place_demo")
    try:
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and node.positions is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.positions is None:
            node.get_logger().error("DIRECT_PICK_PLACE_NO_JOINT_STATE")
            return
        command = node.positions
        node.get_logger().info("DIRECT_PICK_PLACE_START")
        command = _move_gripper(node, "GRIPPER_OPEN", command, GRIPPER_OPEN)
        command = _move_arm(node, "HOME", command, HOME)
        command = _move_arm(node, "PICK_SAFE", command, PICK_SAFE)
        command = _move_arm(node, "PICK_GRASP", command, PICK_GRASP)
        command = _move_gripper(node, "GRIPPER_CLOSE", command, GRIPPER_CLOSED)
        command = _move_arm(node, "LIFT", command, PICK_SAFE)
        command = _move_arm(node, "TRANSIT", command, HOME)
        command = _move_arm(node, "PLACE_SAFE", command, PLACE_SAFE)
        command = _move_arm(node, "PLACE_GRASP", command, PLACE_GRASP)
        command = _move_gripper(node, "GRIPPER_OPEN", command, GRIPPER_OPEN)
        command = _move_arm(node, "RETREAT", command, PLACE_SAFE)
        _move_arm(node, "RETURN_HOME", command, HOME)
        node.get_logger().info("DIRECT_PICK_PLACE_SUCCESS")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
