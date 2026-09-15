from __future__ import annotations

import copy
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desktop_sorter import (
    Detection,
    GridObservation,
    plan_candidates,
    point_to_grid,
    summarize_grid,
    validate_config,
)


def detection(class_name: str, confidence: float = 0.9) -> Detection:
    class_ids = {"cup": 0, "mouse": 1, "bottle": 2}
    return Detection(class_ids[class_name], class_name, confidence, (0, 0, 10, 10), (0.2, 0.2))


def observation(
    grid_id: str,
    state: str,
    class_name: str | None = None,
    confidence: float = 0.0,
) -> GridObservation:
    return GridObservation(grid_id, state, class_name, confidence, 6, 1.0)


class SorterLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "sorter_config.json").read_text(encoding="utf-8"))

    def test_config_is_valid_for_dry_run(self) -> None:
        validate_config(self.config, require_calibrated=False)

    def test_uncalibrated_config_blocks_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "calibrated"):
            validate_config(self.config, require_calibrated=True)

    def test_point_maps_to_exact_grid(self) -> None:
        self.assertEqual(point_to_grid((0.2, 0.7), self.config["grids"]), "G1")
        self.assertEqual(point_to_grid((0.25, 0.2), self.config["grids"]), "G4")
        self.assertIsNone(point_to_grid((0.34, 0.5), self.config["grids"]))

    def test_stable_cup_is_candidate(self) -> None:
        observation = summarize_grid(
            "G1",
            [detection("cup") for _ in range(5)] + [None],
            {"cup", "bottle"},
            min_hits=5,
            min_vote_share=0.7,
        )
        self.assertEqual(observation.state, "candidate")
        self.assertEqual(observation.class_name, "cup")

    def test_mouse_is_logged_as_unsupported(self) -> None:
        observation = summarize_grid(
            "G2",
            [detection("mouse") for _ in range(6)],
            {"cup", "bottle"},
            min_hits=5,
            min_vote_share=0.7,
        )
        self.assertEqual(observation.state, "unsupported")

    def test_conflicting_votes_are_ambiguous(self) -> None:
        observation = summarize_grid(
            "G3",
            [detection("cup") for _ in range(3)] + [detection("bottle") for _ in range(3)],
            {"cup", "bottle"},
            min_hits=3,
            min_vote_share=0.7,
        )
        self.assertEqual(observation.state, "ambiguous")

    def test_execution_requires_arm_points(self) -> None:
        config = copy.deepcopy(self.config)
        config["calibrated"] = True
        with self.assertRaisesRegex(ValueError, "arm_pregrasp_mm"):
            validate_config(config, require_calibrated=True)

    def test_front_row_has_priority_over_higher_confidence_rear_row(self) -> None:
        observations = [
            observation("G1", "candidate", "cup", 0.60),
            observation("G2", "empty"),
            observation("G3", "candidate", "bottle", 0.70),
            observation("G4", "candidate", "bottle", 0.99),
            observation("G5", "empty"),
            observation("G6", "empty"),
        ]
        candidates, blocked = plan_candidates(observations, self.config["grids"], Counter(), 1)
        self.assertEqual([item.grid_id for item in candidates], ["G1", "G3"])
        self.assertEqual(blocked[0]["grid_id"], "G4")

    def test_rear_grid_is_released_after_front_lane_is_empty(self) -> None:
        observations = [
            observation("G1", "empty"),
            observation("G2", "empty"),
            observation("G3", "empty"),
            observation("G4", "candidate", "cup", 0.90),
            observation("G5", "empty"),
            observation("G6", "empty"),
        ]
        candidates, blocked = plan_candidates(observations, self.config["grids"], Counter(), 1)
        self.assertEqual([item.grid_id for item in candidates], ["G4"])
        self.assertEqual(blocked, [])

    def test_mouse_in_front_lane_blocks_rear_drive_path(self) -> None:
        observations = [
            observation("G1", "unsupported", "mouse", 0.95),
            observation("G2", "empty"),
            observation("G3", "empty"),
            observation("G4", "candidate", "bottle", 0.90),
            observation("G5", "empty"),
            observation("G6", "empty"),
        ]
        candidates, blocked = plan_candidates(observations, self.config["grids"], Counter(), 1)
        self.assertEqual(candidates, [])
        self.assertEqual(
            blocked,
            [{"grid_id": "G4", "blocked_by": "G1", "blocker_state": "unsupported"}],
        )

    def test_bin_turn_directions_match_left_and_right(self) -> None:
        self.assertEqual(self.config["bins"]["cup"]["chassis_to_bin"][0]["z_deg"], 90.0)
        self.assertEqual(self.config["bins"]["bottle"]["chassis_to_bin"][0]["z_deg"], -90.0)

    def test_row_edge_gap_is_seven_centimeters(self) -> None:
        front_x = self.config["grids"][0]["ground_center_m"]["x_forward_m"]
        rear_x = self.config["grids"][3]["ground_center_m"]["x_forward_m"]
        square_size = self.config["layout"]["grid_square_size_m"]
        self.assertAlmostEqual(rear_x - front_x - square_size, 0.07)


if __name__ == "__main__":
    unittest.main()
