"""Classify the requested fixed grid with the experiment-one YOLO model."""

from __future__ import annotations

from typing import Mapping, Sequence


def crop_fixed_roi(frame, region: Sequence[float]):
    """Crop a configured ROI and return (crop, x_offset, y_offset)."""
    height, width = frame.shape[:2]
    left, top, right, bottom = (int(round(value)) for value in region)
    left = max(0, min(width, left))
    right = max(0, min(width, right))
    top = max(0, min(height, top))
    bottom = max(0, min(height, bottom))
    if right <= left or bottom <= top:
        raise ValueError("fixed vision ROI is empty")
    return frame[top:bottom, left:right], left, top


def best_yolo_detection(result, class_names: Mapping, confidence_threshold: float) -> dict | None:
    """Return only the highest-confidence class decision from one YOLO result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None
    best = None
    for box in boxes:
        confidence = float(box.conf.item())
        if confidence < float(confidence_threshold):
            continue
        class_index = int(box.cls.item())
        xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
        candidate = {
            "class_id": str(class_names[class_index]),
            "confidence": confidence,
            "xyxy": xyxy,
        }
        if best is None or candidate["confidence"] > best["confidence"]:
            best = candidate
    return best


def majority_class(votes: Sequence[dict | None], minimum_votes: int) -> dict | None:
    """Return a unique class winner from frame-level fixed-ROI decisions."""
    if minimum_votes <= 0:
        raise ValueError("minimum_votes must be positive")
    valid_votes = [vote for vote in votes if vote is not None]
    if not valid_votes:
        return None
    counts: dict[str, int] = {}
    for vote in valid_votes:
        class_id = str(vote["class_id"])
        counts[class_id] = counts.get(class_id, 0) + 1
    highest = max(counts.values())
    winners = [class_id for class_id, count in counts.items() if count == highest]
    if highest < minimum_votes or len(winners) != 1:
        return None
    class_id = winners[0]
    confidences = [
        float(vote["confidence"])
        for vote in valid_votes
        if str(vote["class_id"]) == class_id
    ]
    return {
        "class_id": class_id,
        "confidence": sum(confidences) / len(confidences),
        "vote_count": highest,
        "frame_results": list(votes),
    }


def classify_grid_frames(
    model,
    frames: Sequence,
    region: Sequence[float],
    confidence_threshold: float,
    image_size: int,
    device: str,
    minimum_votes: int = 4,
) -> dict | None:
    """Classify one fixed ROI across buffered frames and vote on its class."""
    frame_results = []
    for frame in frames:
        crop, _, _ = crop_fixed_roi(frame, region)
        result = model.predict(
            source=crop,
            conf=confidence_threshold,
            imgsz=image_size,
            device=device,
            verbose=False,
        )[0]
        best = best_yolo_detection(result, result.names, confidence_threshold)
        frame_results.append(best)
    voted = majority_class(frame_results, minimum_votes)
    if voted is None:
        return None
    return voted


def classify_grid_batch(
    model,
    frames: Sequence,
    grid_regions: Mapping[str, Sequence[float]],
    grid_order: Sequence[str],
    confidence_threshold: float,
    image_size: int,
    device: str,
    minimum_votes: int = 4,
) -> dict[str, dict | None]:
    """Classify every configured fixed ROI over the same buffered frames."""
    return {
        str(grid_id): classify_grid_frames(
            model,
            frames,
            grid_regions[str(grid_id)],
            confidence_threshold,
            image_size,
            device,
            minimum_votes,
        )
        for grid_id in grid_order
    }


def main(args=None) -> None:
    import json
    from collections import deque
    from pathlib import Path

    import cv2
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
    from ultralytics import YOLO

    class YoloClassifierNode(Node):
        def __init__(self) -> None:
            super().__init__("yolo_classifier_node")
            self.declare_parameter("model_path", "models/pencil_tennis_yolo26n_best.pt")
            self.declare_parameter("image_topic", "/camera/image_raw")
            self.declare_parameter("request_topic", "/mecharm/vision_request")
            self.declare_parameter("detection_topic", "/mecharm/detections")
            self.declare_parameter("confidence_threshold", 0.05)
            self.declare_parameter("image_size", 640)
            self.declare_parameter("device", "cpu")
            self.declare_parameter("config_path", "")
            self.declare_parameter("batch_frame_count", 6)
            self.declare_parameter("batch_minimum_votes", 4)
            self.model_path = str(self.get_parameter("model_path").value)
            if not Path(self.model_path).is_file():
                raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
            self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
            self.image_size = int(self.get_parameter("image_size").value)
            self.device = str(self.get_parameter("device").value)
            self.batch_frame_count = int(self.get_parameter("batch_frame_count").value)
            self.batch_minimum_votes = int(self.get_parameter("batch_minimum_votes").value)
            if self.batch_frame_count <= 0 or self.batch_minimum_votes <= 0:
                raise ValueError("batch frame count and minimum votes must be positive")
            self.grid_regions = self._read_grid_regions()
            self.grid_order = tuple(self.grid_regions)
            if self._configured_confidence is not None:
                self.confidence_threshold = self._configured_confidence
            self.model = YOLO(self.model_path)
            self.bridge = CvBridge()
            self.latest_image = None
            self.latest_header = None
            self.frame_buffer = deque(maxlen=self.batch_frame_count)
            self.image_sub = self.create_subscription(
                Image,
                str(self.get_parameter("image_topic").value),
                self._on_image,
                10,
            )
            self.request_sub = self.create_subscription(
                String,
                str(self.get_parameter("request_topic").value),
                self._on_request,
                10,
            )
            self.publisher = self.create_publisher(
                Detection2DArray,
                str(self.get_parameter("detection_topic").value),
                10,
            )
            self.get_logger().info(f"loading fixed-ROI YOLO model: {self.model_path}")

        def _read_grid_regions(self) -> dict[str, tuple[float, ...]]:
            self._configured_confidence = None
            config_path = str(self.get_parameter("config_path").value)
            if not config_path:
                return {}
            import yaml

            with open(config_path, encoding="utf-8") as stream:
                document = yaml.safe_load(stream) or {}
            parameters = document.get("sorting_task", {}).get("ros__parameters", {})
            yolo_parameters = document.get("yolo_classifier", {}).get("ros__parameters", {})
            if "confidence_threshold" in yolo_parameters:
                self._configured_confidence = float(yolo_parameters["confidence_threshold"])
            return {
                str(grid_id): tuple(float(value) for value in region)
                for grid_id, region in parameters.get("grid_regions", {}).items()
            }

        def _on_image(self, message: Image) -> None:
            self.latest_image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            self.latest_header = message.header
            self.frame_buffer.append((self.latest_image.copy(), message.header))

        def _publish_unknown(self, grid_id: str) -> None:
            self.get_logger().warning(
                f"YOLO UNKNOWN grid={grid_id}: no image frame available"
            )
            message = Detection2DArray()
            if self.latest_header is not None:
                message.header = self.latest_header
            detection = Detection2D()
            detection.id = grid_id
            detection.bbox.size_x = 1.0
            detection.bbox.size_y = 1.0
            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = "unknown"
            result.hypothesis.score = 0.0
            detection.results.append(result)
            message.detections.append(detection)
            self.publisher.publish(message)

        def _on_request(self, request: String) -> None:
            grid_id = str(request.data).strip()
            if grid_id.upper() == "ALL":
                self._on_batch_request()
                return
            if grid_id not in self.grid_regions or self.latest_image is None:
                self._publish_unknown(grid_id)
                return
            crop, offset_x, offset_y = crop_fixed_roi(
                self.latest_image, self.grid_regions[grid_id]
            )
            result = self.model.predict(
                source=crop,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )[0]
            names = result.names
            best = best_yolo_detection(result, names, self.confidence_threshold)
            message = Detection2DArray()
            if self.latest_header is not None:
                message.header = self.latest_header
            detection = Detection2D()
            detection.id = grid_id
            if best is None:
                self.get_logger().warning(
                    f"YOLO UNKNOWN grid={grid_id}: no detection in ROI "
                    f"{self.grid_regions[grid_id]} frame_shape={self.latest_image.shape} "
                    f"confidence_threshold={self.confidence_threshold:.3f}"
                )
                left, top, right, bottom = self.grid_regions[grid_id]
                detection.bbox.center.position.x = (left + right) / 2.0
                detection.bbox.center.position.y = (top + bottom) / 2.0
                detection.bbox.size_x = 1.0
                detection.bbox.size_y = 1.0
                result_message = ObjectHypothesisWithPose()
                result_message.hypothesis.class_id = "unknown"
                result_message.hypothesis.score = 0.0
                detection.results.append(result_message)
            else:
                self.get_logger().info(
                    f"YOLO CLASSIFIED grid={grid_id} class={best['class_id']} "
                    f"confidence={best['confidence']:.3f}"
                )
                x1, y1, x2, y2 = best["xyxy"]
                detection.bbox.center.position.x = (x1 + x2) / 2.0 + offset_x
                detection.bbox.center.position.y = (y1 + y2) / 2.0 + offset_y
                detection.bbox.size_x = max(1.0, x2 - x1)
                detection.bbox.size_y = max(1.0, y2 - y1)
                result_message = ObjectHypothesisWithPose()
                result_message.hypothesis.class_id = best["class_id"]
                result_message.hypothesis.score = best["confidence"]
                detection.results.append(result_message)
            message.detections.append(detection)
            self.publisher.publish(message)

        def _on_batch_request(self) -> None:
            if len(self.frame_buffer) < self.batch_frame_count:
                self.get_logger().warning(
                    f"YOLO BATCH UNKNOWN: need {self.batch_frame_count} frames, "
                    f"have {len(self.frame_buffer)}"
                )
                self._publish_batch({grid_id: None for grid_id in self.grid_order})
                return
            frames = [frame for frame, _header in self.frame_buffer]
            decisions = classify_grid_batch(
                self.model,
                frames,
                self.grid_regions,
                self.grid_order,
                self.confidence_threshold,
                self.image_size,
                self.device,
                self.batch_minimum_votes,
            )
            for grid_id, decision in decisions.items():
                if decision is None:
                    self.get_logger().warning(
                        f"YOLO BATCH grid={grid_id} UNKNOWN "
                        f"votes={self.batch_frame_count} "
                        f"minimum={self.batch_minimum_votes}"
                    )
                else:
                    self.get_logger().info(
                        f"YOLO BATCH grid={grid_id} class={decision['class_id']} "
                        f"votes={decision['vote_count']}/{self.batch_frame_count} "
                        f"confidence={decision['confidence']:.3f}"
                    )
            self._publish_batch(decisions)

        def _publish_batch(self, decisions: Mapping[str, dict | None]) -> None:
            message = Detection2DArray()
            if self.latest_header is not None:
                message.header = self.latest_header
            for grid_id in self.grid_order:
                detection = Detection2D()
                detection.id = grid_id
                left, top, right, bottom = self.grid_regions[grid_id]
                detection.bbox.center.position.x = (left + right) / 2.0
                detection.bbox.center.position.y = (top + bottom) / 2.0
                decision = decisions.get(grid_id)
                result_message = ObjectHypothesisWithPose()
                if decision is None:
                    result_message.hypothesis.class_id = "unknown"
                    result_message.hypothesis.score = 0.0
                    detection.bbox.size_x = 1.0
                    detection.bbox.size_y = 1.0
                else:
                    result_message.hypothesis.class_id = decision["class_id"]
                    result_message.hypothesis.score = decision["confidence"]
                    detection.bbox.size_x = max(1.0, right - left)
                    detection.bbox.size_y = max(1.0, bottom - top)
                detection.results.append(result_message)
                message.detections.append(detection)
            self.publisher.publish(message)

    rclpy.init(args=args)
    node = YoloClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
