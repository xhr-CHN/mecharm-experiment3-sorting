from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from mecharm_pick_place.yolo_classifier_node import classify_grid_frames, majority_class


def test_majority_class_requires_four_of_six_votes():
    votes = [
        {"class_id": "pencil", "confidence": 0.60},
        {"class_id": "pencil", "confidence": 0.70},
        {"class_id": "pencil", "confidence": 0.80},
        {"class_id": "pencil", "confidence": 0.90},
        {"class_id": "tennis_ball", "confidence": 0.95},
        None,
    ]

    result = majority_class(votes, minimum_votes=4)

    assert result["class_id"] == "pencil"
    assert result["vote_count"] == 4
    assert result["confidence"] == 0.75
    assert len(result["frame_results"]) == 6


def test_majority_class_rejects_three_three_tie():
    votes = [
        {"class_id": "pencil", "confidence": 0.8},
        {"class_id": "pencil", "confidence": 0.8},
        {"class_id": "pencil", "confidence": 0.8},
        {"class_id": "tennis_ball", "confidence": 0.9},
        {"class_id": "tennis_ball", "confidence": 0.9},
        {"class_id": "tennis_ball", "confidence": 0.9},
    ]

    assert majority_class(votes, minimum_votes=4) is None


def test_classify_grid_frames_rejects_six_unknown_frames():
    class EmptyModel:
        def predict(self, **_kwargs):
            return [type("Result", (), {"boxes": None, "names": {}})()]

    frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(6)]

    assert classify_grid_frames(EmptyModel(), frames, [0, 0, 20, 20], 0.05, 640, "cpu") is None
