from types import SimpleNamespace

from mecharm_pick_place.sorting_task_node import SortingControllerCore, _record_from_message
from mecharm_pick_place.sorting_types import DetectionRecord


def fixture_config():
    return {
        "grid_centers": {
            "G1": [0.2, 0.0, 0.05],
            "G2": [0.0, 0.2, 0.05],
        },
        "grid_regions": {
            "G1": [0, 0, 100, 100],
            "G2": [100, 0, 200, 100],
        },
        "class_bins": {
            "tennis_ball": [0.2, -0.1, 0.05],
            "pencil": [0.3, -0.1, 0.05],
        },
    }


def detection(object_id, class_id, center_x, center_y, confidence=0.9):
    return DetectionRecord(object_id, class_id, confidence, center_x, center_y, 20, 20)


def test_controller_selects_class_bin_from_detection():
    controller = SortingControllerCore(config=fixture_config())
    controller.receive_detections([detection("obj1", "tennis_ball", 50, 50)])
    target = controller.select_next_target()
    assert target.grid_id == "G1"
    assert target.bin_position == (0.2, -0.1, 0.05)


def test_controller_skips_unknown_class_without_motion_command():
    controller = SortingControllerCore(config=fixture_config())
    controller.receive_detections([detection("obj2", "unknown", 150, 50)])
    assert controller.select_next_target() is None
    assert controller.motion_commands == []
    assert controller.results[-1].error_code == "UNKNOWN_CLASS"


def test_controller_records_empty_grid():
    controller = SortingControllerCore(config=fixture_config())
    controller.receive_detections([])
    controller.finish_scan()
    assert controller.results[-1].error_code == "EMPTY_GRID"


def test_controller_deduplicates_grid_and_orders_by_confidence():
    controller = SortingControllerCore(config=fixture_config())
    controller.receive_detections(
        [
            detection("obj2", "pencil", 50, 50, confidence=0.8),
            detection("obj1", "tennis_ball", 50, 50, confidence=0.9),
            detection("obj3", "pencil", 150, 50, confidence=0.7),
        ]
    )
    first = controller.select_next_target()
    assert first.object_id == "obj1"
    second = controller.select_next_target()
    assert second.object_id == "obj3"
    assert controller.select_next_target() is None


def test_controller_rejects_point_outside_configured_reach():
    config = fixture_config()
    config["grid_centers"]["G2"] = [0.8, 0.0, 0.05]
    config["max_reach_radius"] = 0.38
    controller = SortingControllerCore(config=config)
    controller.receive_detections([detection("obj4", "pencil", 150, 50)])
    assert controller.select_next_target() is None
    assert controller.results[-1].error_code == "UNREACHABLE"


def test_fixed_grid_target_ignores_detection_bbox_position():
    controller = SortingControllerCore(config=fixture_config())
    target = controller.target_for_fixed_grid("G2", "pencil", 0.88, "pencil_at_g2")
    assert target.object_id == "pencil_at_g2"
    assert target.pick_position == (0.0, 0.2, 0.05)
    assert target.bin_position == (0.3, -0.1, 0.05)


def test_zero_sized_yolo_detection_is_ignored_instead_of_crashing():
    message = SimpleNamespace(
        id="G1",
        results=[
            SimpleNamespace(
                hypothesis=SimpleNamespace(class_id="unknown", score=0.0)
            )
        ],
        bbox=SimpleNamespace(
            center=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0)),
            size_x=0.0,
            size_y=0.0,
        ),
    )
    assert _record_from_message(message) is None
