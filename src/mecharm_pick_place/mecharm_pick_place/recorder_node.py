"""Record task events, final results, and joint trajectories."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import shutil

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class PickPlaceRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("pick_place_recorder_node")
        self.declare_parameter("result_root", "results/simulation")
        self.declare_parameter("config_path", "")
        root = Path(str(self.get_parameter("result_root").value)).expanduser().resolve()
        self.run_dir = root / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.csv"
        self.trajectory_path = self.run_dir / "trajectory.csv"
        self.errors_path = self.run_dir / "errors.log"
        self._trajectory_header_written = False

        config_path = str(self.get_parameter("config_path").value)
        if config_path:
            source = Path(config_path).expanduser().resolve()
            if source.is_file():
                shutil.copyfile(source, self.run_dir / "parameters.yaml")
            else:
                self.get_logger().warning(f"parameter snapshot source does not exist: {source}")

        with self.summary_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                ["attempt_id", "success", "state", "error_code", "message", "timestamp"]
            )

        self.create_subscription(String, "/mecharm/task_status", self._record_event, 50)
        self.create_subscription(String, "/mecharm/task_result", self._record_result, 10)
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            JointState, "/joint_states", self._record_joint_state, state_qos
        )
        self.get_logger().info(f"recording experiment results in {self.run_dir}")

    def _record_event(self, msg: String) -> None:
        try:
            json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("ignored malformed task status JSON")
            return
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(msg.data + "\n")

    def _record_result(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("ignored malformed task result JSON")
            return
        with self.summary_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [
                    data.get("attempt_id"),
                    data.get("success"),
                    data.get("state"),
                    data.get("error_code"),
                    data.get("message"),
                    data.get("timestamp"),
                ]
            )
        if not bool(data.get("success")):
            with self.errors_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"{data.get('timestamp')} attempt={data.get('attempt_id')} "
                    f"code={data.get('error_code')} message={data.get('message')}\n"
                )

    def _record_joint_state(self, msg: JointState) -> None:
        positions = list(msg.position)
        if len(msg.name) != len(positions):
            self.get_logger().warning("ignored JointState with mismatched names and positions")
            return
        timestamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        mode = "a" if self._trajectory_header_written else "w"
        with self.trajectory_path.open(mode, newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if not self._trajectory_header_written:
                writer.writerow(["timestamp", *msg.name])
                self._trajectory_header_written = True
            writer.writerow([timestamp, *positions])


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickPlaceRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
