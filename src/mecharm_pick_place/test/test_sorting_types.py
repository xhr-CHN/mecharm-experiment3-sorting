import math

import pytest

from mecharm_pick_place.sorting_types import (
    DetectionRecord,
    GridCell,
    bin_for_class,
    find_grid_for_pixel,
    ring_grid_centers,
)


def test_ring_has_six_cells_around_base():
    cells = ring_grid_centers(center=(0.0, 0.0), radius=0.20)
    assert tuple(cells) == ("G1", "G2", "G3", "G4", "G5", "G6")
    assert all(
        math.isclose(math.hypot(point.x, point.y), 0.20, abs_tol=1e-6)
        for point in cells.values()
    )


def test_bbox_center_maps_to_ring_cell():
    regions = {"G1": (0, 0, 100, 100), "G2": (100, 0, 200, 100)}
    assert find_grid_for_pixel(50, 50, regions) == "G1"
    assert find_grid_for_pixel(150, 50, regions) == "G2"


def test_pixel_outside_grid_is_unknown():
    assert find_grid_for_pixel(300, 300, {"G1": (0, 0, 100, 100)}) is None


def test_class_maps_to_only_configured_bins():
    bins = {"tennis_ball": (0.2, -0.1, 0.05), "pencil": (0.3, -0.1, 0.05)}
    assert bin_for_class("tennis_ball", bins) == bins["tennis_ball"]
    assert bin_for_class("unknown", bins) is None


def test_invalid_grid_region_is_rejected():
    with pytest.raises(ValueError, match="four values"):
        find_grid_for_pixel(1, 1, {"G1": (0, 0, 1)})


def test_detection_record_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="positive"):
        DetectionRecord("obj1", "pencil", 0.9, 1.0, 1.0, 0.0, 2.0)
