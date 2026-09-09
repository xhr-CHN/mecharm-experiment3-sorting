"""Automatic detection-driven sorting controller and ROS 2 adapter."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Iterable, Mapping

from .sorting_types import (
    DetectionRecord,
    SortingResult,
    SortingTarget,
    bin_for_class,
    find_grid_for_pixel,
)


class SortingControllerCore:
    """Pure target selection and result policy for a sorting batch."""

    def __init__(self, config: Mapping) -> None:
        self.config = config
        self._detections: list[DetectionRecord] = []
        self._used_grids: set[str] = set()
        self.results: list[SortingResult] = []
        self.motion_commands: list[dict] = []

    def receive_detections(self, detections: Iterable[DetectionRecord]) -> None:
        self._detections = sorted(
            list(detections),
            key=lambda item: (-item.confidence, item.object_id),
        )

    def select_next_target(self) -> SortingTarget | None:
        regions = self.config.get("grid_regions", {})
        centers = self.config.get("grid_centers", {})
        bins = self.config.get("class_bins", {})
        for detection in self._detections:
            grid_id = find_grid_for_pixel(
                detection.center_x, detection.center_y, regions
            )
            if grid_id is None:
                self._record_failure(detection, None, "UNKNOWN_GRID", "detection is outside the ring")
                continue
            if grid_id in self._used_grids:
                continue
            if grid_id not in centers:
                self._record_failure(detection, grid_id, "UNKNOWN_GRID", "grid has no pickup coordinate")
                continue
            pick_position = tuple(float(value) for value in centers[grid_id])
            max_reach = self.config.get("max_reach_radius")
            if max_reach is not None and (pick_position[0] ** 2 + pick_position[1] ** 2) ** 0.5 > float(max_reach):
                self._record_failure(detection, grid_id, "UNREACHABLE", "fixed pickup point exceeds configured reach")
                continue
            bin_position = bin_for_class(detection.class_id, bins)
            if bin_position is None:
                self._record_failure(detection, grid_id, "UNKNOWN_CLASS", f"no bin for {detection.class_id}")
                continue
            self._used_grids.add(grid_id)
            return SortingTarget(
                object_id=detection.object_id,
                class_id=detection.class_id,
                grid_id=grid_id,
                pick_position=pick_position,
                bin_position=bin_position,
            )
        return None

    def finish_scan(self) -> None:
        if not self._detections:
            self.results.append(
                SortingResult(None, None, None, None, False, "EMPTY_GRID", "no detections in the scan")
            )

    def target_for_fixed_grid(
        self,
        grid_id: str,
        class_id: str,
        confidence: float,
        object_id: str | None = None,
    ) -> SortingTarget | None:
        """Build a target from a fixed grid; YOLO geometry is not an input."""
        centers = self.config.get("grid_centers", {})
        bins = self.config.get("class_bins", {})
        if grid_id not in centers:
            self.record_fixed_failure(grid_id, "UNKNOWN_GRID", "grid has no pickup coordinate")
            return None
        bin_position = bin_for_class(class_id, bins)
        if bin_position is None:
            self.record_fixed_failure(grid_id, "UNKNOWN_CLASS", f"no bin for {class_id}")
            return None
        pick_position = tuple(float(value) for value in centers[grid_id])
        max_reach = self.config.get("max_reach_radius")
        if max_reach is not None and (pick_position[0] ** 2 + pick_position[1] ** 2) ** 0.5 > float(max_reach):
            self.record_fixed_failure(grid_id, "UNREACHABLE", "fixed pickup point exceeds configured reach")
            return None
        self._used_grids.add(grid_id)
        return SortingTarget(
            object_id=object_id or f"{grid_id}_object",
            class_id=class_id,
            grid_id=grid_id,
            pick_position=pick_position,
            bin_position=bin_position,
        )

    def record_fixed_failure(self, grid_id: str, code: str, message: str) -> SortingResult:
        result = SortingResult(None, None, grid_id, None, False, code, message)
        self.results.append(result)
        return result

    def record_success(self, target: SortingTarget, message: str = "placed and verified") -> SortingResult:
        result = SortingResult(
            target.object_id,
            target.class_id,
            target.grid_id,
            target.bin_position,
            True,
            "NONE",
            message,
        )
        self.results.append(result)
        return result

    def record_failure(self, target: SortingTarget, code: str, message: str) -> SortingResult:
        result = SortingResult(
            target.object_id,
            target.class_id,
            target.grid_id,
            target.bin_position,
            False,
            code,
            message,
        )
        self.results.append(result)
        return result

    def _record_failure(
        self,
        detection: DetectionRecord,
        grid_id: str | None,
        code: str,
        message: str,
    ) -> None:
        self.results.append(
            SortingResult(
                detection.object_id,
                detection.class_id,
                grid_id,
                None,
                False,
                code,
                message,
            )
        )


def _load_config(path: str) -> dict:
    import yaml

    with Path(path).expanduser().resolve().open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return data.get("sorting_task", {}).get("ros__parameters", data)


def _record_from_message(message) -> DetectionRecord | None:
    if not message.results:
        return None
    hypothesis = message.results[0].hypothesis
    try:
        return DetectionRecord(
            object_id=str(message.id),
            class_id=str(hypothesis.class_id),
            confidence=float(hypothesis.score),
            center_x=float(message.bbox.center.position.x),
            center_y=float(message.bbox.center.position.y),
            width=float(message.bbox.size_x),
            height=float(message.bbox.size_y),
        )
    except (TypeError, ValueError):
        return None


def batch_classes_by_grid(
    records: Iterable[DetectionRecord], grid_order: Iterable[str]
) -> dict[str, tuple[str, float]] | None:
    """Index one complete batch of fixed-grid class decisions."""
    expected = tuple(str(grid_id) for grid_id in grid_order)
    indexed: dict[str, tuple[str, float]] = {}
    for record in records:
        grid_id = str(record.object_id)
        if grid_id in expected and grid_id not in indexed:
            indexed[grid_id] = (str(record.class_id), float(record.confidence))
    if set(indexed) != set(expected):
        return None
    return {grid_id: indexed[grid_id] for grid_id in expected}


def main(args=None) -> None:
    import rclpy
    import math
    import numpy as np
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from vision_msgs.msg import Detection2DArray
    from .cartesian_direct_pick_place import CartesianDirectPickPlace
    from .direct_pick_place_demo import HOME, _move, _move_arm

    class SortingTaskNode(CartesianDirectPickPlace):
        def __init__(self) -> None:
            super().__init__()
            self.external_executor_spins = True
            self.declare_parameter("config_path", "")
            self.declare_parameter("auto_start", True)
            self.declare_parameter("result_root", "results/experiment3")
            config_path = str(self.get_parameter("config_path").value)
            self.core = SortingControllerCore(_load_config(config_path)) if config_path else SortingControllerCore({})
            self.active = bool(self.get_parameter("auto_start").value)
            self.running = False
            self.grid_order = tuple(
                self.core.config.get(
                    "fixed_grid_order", tuple(self.core.config.get("grid_centers", {}))
                )
            )
            self.object_ids_by_grid = {
                str(item["grid_id"]): str(item["object_id"])
                for item in self.core.config.get("objects", [])
            }
            self.vision_timeout = float(self.core.config.get("vision_timeout_sec", 8.0))
            self.vision_request_pub = self.create_publisher(
                String,
                str(self.core.config.get("vision_request_topic", "/mecharm/vision_request")),
                10,
            )
            self._batch_initialized = False
            self._grid_index = 0
            self._waiting_batch = False
            self._classified_by_grid = None
            self._vision_deadline = 0.0
            self._feedback_deadline = 0.0
            self._command = None
            self._returned_home = False
            self.status_pub = self.create_publisher(String, "/mecharm/sorting_status", 20)
            self.result_pub = self.create_publisher(String, "/mecharm/sorting_result", 20)
            self.create_subscription(Detection2DArray, "/mecharm/detections", self._on_detections, 10)
            self.create_service(Trigger, "/mecharm/sorting_start", self._start)
            self.create_service(Trigger, "/mecharm/sorting_stop", self._stop)
            self.batch_timer = self.create_timer(0.1, self._batch_tick)
            self.result_path = Path(str(self.get_parameter("result_root").value)).expanduser().resolve()
            self.result_path.mkdir(parents=True, exist_ok=True)
            self.detections_path = self.result_path / "detections.jsonl"
            self.results_path = self.result_path / "sorting_results.jsonl"
            self.events_path = self.result_path / "task_events.jsonl"

        def _start(self, _request, response):
            self.active = True
            self.running = False
            self.core = SortingControllerCore(self.core.config)
            self.grid_order = tuple(
                self.core.config.get(
                    "fixed_grid_order", tuple(self.core.config.get("grid_centers", {}))
                )
            )
            self._batch_initialized = False
            self._grid_index = 0
            self._waiting_batch = False
            self._classified_by_grid = None
            self._feedback_deadline = 0.0
            self._returned_home = False
            self._publish_status("WAIT_DETECTIONS", True, "autonomous sorting started")
            response.success = True
            response.message = "sorting started without a manually selected target"
            return response

        def _stop(self, _request, response):
            self.active = False
            self._waiting_batch = False
            self._publish_status("SAFE_STOP", False, "operator requested stop")
            response.success = True
            response.message = "sorting stopped"
            return response

        def _on_detections(self, message: Detection2DArray) -> None:
            if not self.active or not self._waiting_batch:
                return
            records = []
            for item in message.detections:
                record = _record_from_message(item)
                if record is not None:
                    records.append(record)
            classified = batch_classes_by_grid(records, self.grid_order)
            if classified is None:
                self.get_logger().warning(
                    f"YOLO BATCH incomplete: expected grids {self.grid_order}"
                )
                return
            self._write_jsonl(
                self.detections_path,
                {
                    "timestamp": time.time(),
                    "mode": "batch",
                    "grid_ids": list(self.grid_order),
                    "count": len(records),
                    "detections": [asdict(record) for record in records],
                },
            )
            self._classified_by_grid = classified
            self._waiting_batch = False
            self._publish_status(
                "VISION_BATCH_READY", True,
                f"locked classes for {len(self.grid_order)} fixed grids",
            )

        def _batch_tick(self) -> None:
            if not self.active or self.running:
                return
            now = time.monotonic()
            if self.positions is None:
                if self._feedback_deadline == 0.0:
                    self._feedback_deadline = now + 8.0
                elif now >= self._feedback_deadline:
                    self._publish_status("SAFE_STOP", False, "no joint feedback from Isaac")
                    self.active = False
                return
            if not self._batch_initialized:
                self.running = True
                try:
                    self._command = self.move_gripper(
                        "INITIAL_GRIPPER_OPEN", self.positions, self.get_parameter("gripper_open").value
                    )
                    self._command = _move_arm(self, "HOME", self._command, HOME)
                    self._batch_initialized = True
                except (RuntimeError, ValueError) as exc:
                    self._publish_status("SAFE_STOP", False, str(exc))
                    self.active = False
                finally:
                    self.running = False
                return
            if self._waiting_batch:
                if self._classified_by_grid is None:
                    if now < self._vision_deadline:
                        return
                    self._publish_status(
                        "SAFE_STOP", False,
                        "YOLO batch classification timed out before all six grids were classified",
                    )
                    self._waiting_batch = False
                    self.active = False
                    return
                return
            if self._classified_by_grid is not None and any(
                class_id == "unknown"
                for class_id, _confidence in self._classified_by_grid.values()
            ):
                for grid_id, (class_id, _confidence) in self._classified_by_grid.items():
                    if class_id == "unknown":
                        self._publish_result(
                            self.core.record_fixed_failure(
                                grid_id, "UNKNOWN_CLASS", "batch YOLO vote did not reach the minimum"
                            )
                        )
                self._publish_status("SAFE_STOP", False, "YOLO batch contains unknown classes")
                self.active = False
                return
            if self._classified_by_grid is not None and self._grid_index < len(self.grid_order):
                grid_id = str(self.grid_order[self._grid_index])
                class_id, confidence = self._classified_by_grid[grid_id]
                target = self.core.target_for_fixed_grid(
                    grid_id,
                    class_id,
                    confidence,
                    self.object_ids_by_grid.get(grid_id),
                )
                if target is None:
                    if self.core.results:
                        self._publish_result(self.core.results[-1])
                    self._grid_index += 1
                    return
                self.running = True
                try:
                    self._publish_status("PICK", True, f"fixed {grid_id}: YOLO classified {class_id}")
                    self._command = self._execute_target(self._command, target)
                    self._publish_result(self.core.record_success(target, "pick place motion completed"))
                except (RuntimeError, ValueError) as exc:
                    self._publish_result(self.core.record_failure(target, "PICK_FAILED", str(exc)))
                    self._publish_status("SAFE_STOP", False, str(exc))
                    self.active = False
                finally:
                    self.running = False
                self._grid_index += 1
                return
            if self._grid_index >= len(self.grid_order):
                if not self._returned_home:
                    self.running = True
                    try:
                        self._publish_status("FINAL_HOME", True, "returning home after final fixed grid")
                        self._command = _move_arm(self, "RETURN_HOME", self._command, HOME)
                        self._returned_home = True
                    except (RuntimeError, ValueError) as exc:
                        self._publish_status("SAFE_STOP", False, str(exc))
                        self.active = False
                    finally:
                        self.running = False
                    return
                self._publish_status("DONE", True, "fixed grid sequence completed")
                self.active = False
                return
            if self._classified_by_grid is None:
                self._publish_status(
                    "VISION_BATCH_REQUEST", True,
                    f"classifying all fixed grids over {self.core.config.get('batch_frame_count', 6)} frames",
                )
                self._waiting_batch = True
                self._vision_deadline = now + self.vision_timeout
                self.vision_request_pub.publish(String(data="ALL"))

        def _execute_target(self, command, target: SortingTarget):
            import numpy as np

            pregrasp = float(self.get_parameter("pregrasp_clearance").value)
            carry_clearance = float(self.core.config.get("carry_clearance", 0.108))
            pick_close = float(self.get_parameter("pick_close_clearance").value)
            open_position = float(self.get_parameter("gripper_open").value)
            closed_position = float(self.get_parameter("gripper_closed").value)
            home_tool_x = self.solver.forward(command[:6])[:3, 0]
            home_tool_x[2] = 0.0
            home_tool_x /= np.linalg.norm(home_tool_x)
            yaw = math.radians(float(self.get_parameter("grasp_yaw_offset_deg").value))
            c, s = math.cos(yaw), math.sin(yaw)
            grasp_tool_x = np.asarray(
                (c * home_tool_x[0] - s * home_tool_x[1], c * home_tool_x[1] + s * home_tool_x[0], 0.0),
                dtype=float,
            )
            pick = target.pick_position
            place = target.bin_position
            if place is None:
                raise RuntimeError(f"no bin configured for {target.class_id}")
            command = _move(self, "PICK_PREGRASP", command, self.solve("PICK_PREGRASP", pick, pregrasp, command, grasp_tool_x))
            command = _move(self, "PICK_GRASP", command, self.solve("PICK_GRASP", pick, pick_close, command, grasp_tool_x))
            command = self.move_gripper("GRIPPER_CLOSE_ON_OBJECT", command, closed_position)
            command = _move(self, "LIFT", command, self.solve("LIFT", pick, carry_clearance, command, grasp_tool_x))
            command = _move(self, "PLACE_PREGRASP", command, self.solve("PLACE_PREGRASP", place, carry_clearance, command, grasp_tool_x))
            return self.move_gripper("GRIPPER_OPEN_ABOVE_BIN", command, open_position)

        def _publish_status(self, state: str, success: bool, message: str) -> None:
            payload = {"state": state, "success": success, "message": message, "timestamp": time.time()}
            self.get_logger().info(f"{state}: {message}")
            self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
            self._write_jsonl(self.events_path, payload)

        def _publish_result(self, result: SortingResult) -> None:
            payload = asdict(result)
            self.get_logger().info(
                f"RESULT {result.grid_id}: success={result.success} "
                f"class={result.class_id} code={result.error_code}"
            )
            self.result_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
            self._write_jsonl(self.results_path, payload)

        @staticmethod
        def _write_jsonl(path: Path, payload: dict) -> None:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    rclpy.init(args=args)
    node = SortingTaskNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
