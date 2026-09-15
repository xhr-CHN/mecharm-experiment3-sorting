"""Run cup, mouse, and bottle detection from a camera or video stream."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PROJECT_ROOT / ".ultralytics"
try:
    (CONFIG_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
    if not os.access(CONFIG_ROOT / "Ultralytics", os.W_OK):
        raise OSError("Project config directory is not writable")
except OSError:
    CONFIG_ROOT = Path(tempfile.gettempdir()) / "ultralytics_config"
    (CONFIG_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_ROOT))

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "best.pt",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video/stream URL, or gst:<GStreamer pipeline>",
    )
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/Ctrl+C")
    parser.add_argument(
        "--ros2",
        action="store_true",
        help="Publish one JSON message per processed frame through ROS2",
    )
    parser.add_argument("--ros-topic", default="/detections")
    parser.add_argument("--ros-node-name", default="yolo_detector")
    return parser.parse_args()


def open_capture(source_text: str, width: int, height: int) -> cv2.VideoCapture:
    if source_text.startswith("gst:"):
        capture = cv2.VideoCapture(source_text[4:], cv2.CAP_GSTREAMER)
    else:
        source: int | str = int(source_text) if source_text.isdigit() else source_text
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera or stream: {source_text}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.exists():
        raise FileNotFoundError(weights)

    rclpy_module = None
    ros_node_type = None
    ros_string_type = None
    if args.ros2:
        try:
            import rclpy as rclpy_module
            from rclpy.node import Node as RosNode
            from std_msgs.msg import String as RosString
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "ROS2 Python packages are unavailable. Source the ROS2 environment "
                "before running with --ros2."
            ) from error
        ros_node_type = RosNode
        ros_string_type = RosString

    model = YOLO(str(weights))
    required_classes = {"cup", "mouse", "bottle"}
    model_classes = {str(name) for name in model.names.values()}
    missing_classes = required_classes - model_classes
    if missing_classes:
        raise RuntimeError(
            "The selected model is missing required classes: "
            + ", ".join(sorted(missing_classes))
        )
    capture = open_capture(args.source, args.width, args.height)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    output_fps = source_fps if 1.0 <= source_fps <= 240.0 else 30.0

    writer: cv2.VideoWriter | None = None
    csv_handle = None
    csv_writer: csv.DictWriter | None = None
    output_dir = args.output_dir.resolve()
    if args.save:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_dir / "realtime_detected.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (actual_width, actual_height),
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot create output video")
        csv_handle = (output_dir / "detections.csv").open("w", newline="", encoding="utf-8-sig")
        fields = [
            "frame",
            "elapsed_seconds",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
        csv_writer = csv.DictWriter(csv_handle, fieldnames=fields)
        csv_writer.writeheader()

    frame_index = 0
    smoothed_fps = 0.0
    total_inference_ms = 0.0
    class_counts: Counter[str] = Counter()
    published_messages = 0
    started = time.perf_counter()

    ros_node = None
    ros_publisher = None
    if args.ros2:
        assert rclpy_module is not None
        assert ros_node_type is not None
        assert ros_string_type is not None
        rclpy_module.init(args=[])
        ros_node = ros_node_type(args.ros_node_name)
        ros_publisher = ros_node.create_publisher(ros_string_type, args.ros_topic, 10)
        ros_node.get_logger().info(f"Publishing detections on {args.ros_topic}")

    try:
        while True:
            frame_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            total_inference_ms += float(result.speed.get("inference", 0.0))
            annotated = result.plot()
            frame_detections: list[dict[str, object]] = []

            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                scores = result.boxes.conf.detach().cpu().tolist()
                for box, class_id, score in zip(boxes, classes, scores):
                    name = str(result.names[int(class_id)])
                    class_counts[name] += 1
                    detection = {
                        "class_id": int(class_id),
                        "class_name": name,
                        "confidence": float(score),
                        "bbox": {
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                        },
                    }
                    frame_detections.append(detection)
                    if csv_writer is not None:
                        csv_writer.writerow(
                            {
                                "frame": frame_index,
                                "elapsed_seconds": time.perf_counter() - started,
                                "class_id": int(class_id),
                                "class_name": name,
                                "confidence": float(score),
                                "x1": float(box[0]),
                                "y1": float(box[1]),
                                "x2": float(box[2]),
                                "y2": float(box[3]),
                            }
                        )

            elapsed = max(time.perf_counter() - frame_started, 1e-9)
            instantaneous_fps = 1.0 / elapsed
            smoothed_fps = (
                instantaneous_fps
                if frame_index == 0
                else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
            )

            if ros_publisher is not None and ros_node is not None and ros_string_type is not None:
                ros_message = ros_string_type()
                ros_message.data = json.dumps(
                    {
                        "stamp_ns": ros_node.get_clock().now().nanoseconds,
                        "frame": frame_index,
                        "image_width": actual_width,
                        "image_height": actual_height,
                        "fps": smoothed_fps,
                        "detections": frame_detections,
                    },
                    ensure_ascii=False,
                )
                ros_publisher.publish(ros_message)
                assert rclpy_module is not None
                rclpy_module.spin_once(ros_node, timeout_sec=0.0)
                published_messages += 1

            cv2.putText(
                annotated,
                f"FPS: {smoothed_fps:.1f}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(annotated)
            frame_index += 1
            if not args.no_display:
                cv2.imshow("YOLO cup / mouse / bottle", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_index >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if csv_handle is not None:
            csv_handle.close()
        cv2.destroyAllWindows()
        if ros_node is not None:
            ros_node.destroy_node()
            assert rclpy_module is not None
            if rclpy_module.ok():
                rclpy_module.shutdown()

    wall_seconds = time.perf_counter() - started
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "weights": str(weights),
        "settings": {
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "image_size": args.imgsz,
            "device": args.device,
        },
        "frames": frame_index,
        "wall_seconds": wall_seconds,
        "processing_fps": frame_index / wall_seconds if wall_seconds else 0.0,
        "average_inference_ms": total_inference_ms / frame_index if frame_index else 0.0,
        "box_detections_by_class": dict(class_counts),
        "ros2": {
            "enabled": args.ros2,
            "topic": args.ros_topic if args.ros2 else None,
            "messages_published": published_messages,
        },
    }
    if args.save:
        (output_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
