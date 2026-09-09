from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "simulation" / "isaac"))
from experiment3_camera_stream import normalize_rgb_frame
from mecharm_pick_place.yolo_classifier_node import best_yolo_detection, crop_fixed_roi


def test_fixed_roi_crop_returns_offset_without_inferencing_position():
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    crop, offset_x, offset_y = crop_fixed_roi(frame, (5, 4, 15, 12))
    assert crop.shape == (8, 10, 3)
    assert (offset_x, offset_y) == (5, 4)


def test_best_yolo_detection_returns_only_class_and_observation_box():
    box = SimpleNamespace(
        conf=SimpleNamespace(item=lambda: 0.91),
        cls=SimpleNamespace(item=lambda: 1),
        xyxy=[SimpleNamespace(tolist=lambda: [1.0, 2.0, 8.0, 9.0])],
    )
    result = best_yolo_detection(SimpleNamespace(boxes=[box]), {1: "pencil"}, 0.25)
    assert result == {
        "class_id": "pencil",
        "confidence": 0.91,
        "xyxy": (1.0, 2.0, 8.0, 9.0),
    }


def test_camera_stream_normalizes_flat_rgba_buffer():
    flat = np.arange(2 * 3 * 4, dtype=np.uint8)
    frame = normalize_rgb_frame(flat, width=3, height=2)
    assert frame.shape == (2, 3, 3)
    assert frame[0, 0].tolist() == [0, 1, 2]


def test_camera_stream_rejects_unknown_buffer_shape():
    with np.testing.assert_raises(ValueError):
        normalize_rgb_frame(np.zeros(7, dtype=np.uint8), width=3, height=2)


def test_unknown_yolo_result_uses_positive_placeholder_box():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "mecharm_pick_place"
        / "mecharm_pick_place"
        / "yolo_classifier_node.py"
    ).read_text(encoding="utf-8")
    assert source.count("detection.bbox.size_x = 1.0") >= 2
    assert source.count("detection.bbox.size_y = 1.0") >= 2
    assert "self._configured_confidence" in source
    assert "YOLO UNKNOWN grid=" in source
