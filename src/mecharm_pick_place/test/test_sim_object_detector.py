from mecharm_pick_place.sim_object_detector import (
    DetectionRecord,
    detection_payload,
    filter_detections,
)


def test_detection_payload_contains_required_fields():
    payload = detection_payload(
        DetectionRecord("obj1", "tennis_ball", 0.94, 120.0, 80.0, 30.0, 30.0)
    )
    assert payload["object_id"] == "obj1"
    assert payload["class_id"] == "tennis_ball"
    assert payload["confidence"] == 0.94
    assert payload["bbox"] == {
        "center_x": 120.0,
        "center_y": 80.0,
        "width": 30.0,
        "height": 30.0,
    }


def test_low_confidence_detection_is_filtered():
    records = filter_detections(
        [DetectionRecord("obj1", "pencil", 0.3, 1, 1, 2, 2)], threshold=0.5
    )
    assert records == []


def test_detection_order_is_confidence_then_object_id():
    records = filter_detections(
        [
            DetectionRecord("obj2", "pencil", 0.8, 1, 1, 2, 2),
            DetectionRecord("obj1", "tennis_ball", 0.9, 1, 1, 2, 2),
        ],
        threshold=0.5,
    )
    assert [record.object_id for record in records] == ["obj1", "obj2"]
