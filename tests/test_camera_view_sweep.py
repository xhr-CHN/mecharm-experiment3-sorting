from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "simulation" / "isaac"))
sys.path.insert(0, str(ROOT / "tools"))

from camera_view_sweep import camera_candidates, is_usable_frame
from evaluate_camera_views import project_world_point, ranking_key, summarize_view


def test_camera_sweep_has_thirteen_unique_views():
    candidates = camera_candidates()
    assert len(candidates) == 13
    assert len({item["view_id"] for item in candidates}) == 13
    assert candidates[0]["position"] == [0.0, 0.0, 1.10]


def test_camera_sweep_rejects_black_frames():
    import numpy as np

    assert not is_usable_frame(np.zeros((480, 848, 3), dtype=np.uint8))
    visible = np.zeros((480, 848, 3), dtype=np.uint8)
    visible[100:200, 100:200] = 200
    assert is_usable_frame(visible)


def test_camera_summary_prioritizes_stable_pencil_recall():
    records = [
        {
            "view_id": "candidate",
            "correct_count": value,
            "pencil_recall": 1.0,
            "tennis_recall": value / 6.0,
            "selected_confidences": [0.8],
            "wrong_count": 0,
        }
        for value in (6, 6, 5, 6, 6)
    ]
    summary = summarize_view(records)
    assert summary["frame_count"] == 5
    assert summary["min_correct"] == 5
    assert summary["mean_pencil_recall"] == 1.0
    assert ranking_key(summary)[0] == 5


def test_vertical_projection_matches_current_fixed_grid_image_locations():
    pixel = project_world_point(
        [0.18, 0.08, 0.025], [0.0, 0.0, 1.10], [0.0, 0.0, 0.0],
        848, 480, 18.0, 20.955,
    )
    assert abs(pixel[0] - 546.0) < 2.0
    assert abs(pixel[1] - 186.0) < 2.0
