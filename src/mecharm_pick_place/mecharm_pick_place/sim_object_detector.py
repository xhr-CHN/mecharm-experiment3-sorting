"""Simulation detector with a ROS vision_msgs-compatible output contract.

The pure conversion/filter functions are usable without a ROS installation. The
ROS node consumes camera frames as a synchronization trigger and publishes the
configured simulation detections for the stable first integration scenario.
"""

from __future__ import annotations

import json
from typing import Iterable

from .sorting_types import DetectionRecord

__all__ = [
    "DetectionRecord",
    "detection_payload",
    "filter_detections",
    "to_detection_message",
]


def detection_payload(record: DetectionRecord) -> dict:
    return {
        "object_id": record.object_id,
        "class_id": record.class_id,
        "confidence": float(record.confidence),
        "bbox": {
            "center_x": float(record.center_x),
            "center_y": float(record.center_y),
            "width": float(record.width),
            "height": float(record.height),
        },
    }


def filter_detections(
    records: Iterable[DetectionRecord], threshold: float
) -> list[DetectionRecord]:
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("confidence threshold must be between 0 and 1")
    return sorted(
        (record for record in records if record.confidence >= threshold),
        key=lambda record: (-record.confidence, record.object_id),
    )


def to_detection_message(record: DetectionRecord):
    """Convert a record to `vision_msgs/Detection2D` lazily at ROS runtime."""
    from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

    message = Detection2D()
    message.id = record.object_id
    message.bbox.center.position.x = float(record.center_x)
    message.bbox.center.position.y = float(record.center_y)
    message.bbox.size_x = float(record.width)
    message.bbox.size_y = float(record.height)
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id = record.class_id
    result.hypothesis.score = float(record.confidence)
    message.results.append(result)
    return message


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from vision_msgs.msg import Detection2DArray

    class SimObjectDetectorNode(Node):
        def __init__(self) -> None:
            super().__init__("sim_object_detector")
            self.declare_parameter("image_topic", "/camera/image_raw")
            self.declare_parameter("detection_topic", "/mecharm/detections")
            self.declare_parameter("confidence_threshold", 0.5)
            self.declare_parameter("scenario", "normal")
            self.declare_parameter("object_records_json", "[]")
            self.declare_parameter("config_path", "")
            self._config = self._read_config()
            if self._config:
                for name, value in self._config.items():
                    if name != "object_records_json":
                        self.set_parameters([rclpy.parameter.Parameter(name, value=value)])
            self.publisher = self.create_publisher(
                Detection2DArray,
                str(self.get_parameter("detection_topic").value),
                10,
            )
            self.records = self._read_records()
            self.subscription = self.create_subscription(
                Image,
                str(self.get_parameter("image_topic").value),
                self._on_image,
                10,
            )
            self.get_logger().info(
                f"simulation detector ready with {len(self.records)} configured records"
            )

        def _read_config(self) -> dict:
            config_path = str(self.get_parameter("config_path").value)
            if not config_path:
                return {}
            import yaml

            with open(config_path, encoding="utf-8") as stream:
                document = yaml.safe_load(stream) or {}
            return document.get("sim_object_detector", {}).get("ros__parameters", {})

        def _read_records(self) -> list[DetectionRecord]:
            raw = str(
                self._config.get(
                    "object_records_json",
                    self.get_parameter("object_records_json").value,
                )
            )
            try:
                entries = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"object_records_json is invalid: {exc}") from exc
            if not isinstance(entries, list):
                raise ValueError("object_records_json must be a list")
            records = [DetectionRecord(**entry) for entry in entries]
            threshold = float(
                self._config.get(
                    "confidence_threshold",
                    self.get_parameter("confidence_threshold").value,
                )
            )
            return filter_detections(records, threshold)

        def _on_image(self, image) -> None:
            scenario = str(self.get_parameter("scenario").value)
            records = [] if scenario == "empty" else self.records
            message = Detection2DArray()
            message.header = image.header
            message.detections = [to_detection_message(record) for record in records]
            self.publisher.publish(message)

    rclpy.init(args=args)
    node = SimObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
