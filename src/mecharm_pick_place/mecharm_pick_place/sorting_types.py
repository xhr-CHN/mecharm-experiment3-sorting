"""Pure data contracts and geometry helpers for experiment three sorting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class GridCell:
    grid_id: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class DetectionRecord:
    object_id: str
    class_id: str
    confidence: float
    center_x: float
    center_y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if float(self.width) <= 0.0 or float(self.height) <= 0.0:
            raise ValueError("detection width and height must be positive")


@dataclass(frozen=True)
class SortingTarget:
    object_id: str
    class_id: str
    grid_id: str
    pick_position: tuple[float, float, float]
    bin_position: tuple[float, float, float] | None


@dataclass(frozen=True)
class SortingResult:
    object_id: str | None
    class_id: str | None
    grid_id: str | None
    bin_position: tuple[float, float, float] | None
    success: bool
    error_code: str
    message: str


def ring_grid_centers(
    center: tuple[float, float],
    radius: float,
    z: float = 0.05,
    start_angle_deg: float = 90.0,
) -> dict[str, GridCell]:
    """Return six evenly spaced pickup cells around the robot base."""
    if radius <= 0.0:
        raise ValueError("ring radius must be positive")
    cx, cy = center
    cells: dict[str, GridCell] = {}
    for index in range(6):
        angle = math.radians(start_angle_deg + index * 60.0)
        cells[f"G{index + 1}"] = GridCell(
            grid_id=f"G{index + 1}",
            x=float(cx + radius * math.cos(angle)),
            y=float(cy + radius * math.sin(angle)),
            z=float(z),
        )
    return cells


def find_grid_for_pixel(
    center_x: float,
    center_y: float,
    regions: Mapping[str, Sequence[float]],
) -> str | None:
    """Return the first configured grid containing a pixel center."""
    for grid_id, region in regions.items():
        if len(region) != 4:
            raise ValueError(f"grid region for {grid_id} must contain four values")
        left, top, right, bottom = (float(value) for value in region)
        if right < left or bottom < top:
            raise ValueError(f"grid region for {grid_id} has reversed bounds")
        if left <= center_x <= right and top <= center_y <= bottom:
            return str(grid_id)
    return None


def bin_for_class(
    class_id: str,
    bins: Mapping[str, Sequence[float]],
) -> tuple[float, float, float] | None:
    """Map only configured classes to a three-dimensional bin position."""
    position = bins.get(class_id)
    if position is None:
        return None
    if len(position) != 3:
        raise ValueError(f"bin position for {class_id} must contain three values")
    return tuple(float(value) for value in position)  # type: ignore[return-value]
