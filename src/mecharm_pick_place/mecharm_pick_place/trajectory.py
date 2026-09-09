"""Pure trajectory validation and interpolation used by the ROS action adapter."""

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class RawPoint:
    positions: tuple[float, ...]
    time_from_start: float


@dataclass(frozen=True)
class NormalizedTrajectory:
    joint_names: tuple[str, ...]
    points: tuple[RawPoint, ...]


def normalize_trajectory(joint_names, points, required_names, limits):
    names = tuple(joint_names)
    required = tuple(required_names)
    points = tuple(points)
    if len(names) != len(set(names)) or set(names) != set(required):
        raise ValueError("goal must contain each required joint exactly once")
    if not points:
        raise ValueError("trajectory is empty")
    source_index = {name: index for index, name in enumerate(names)}
    normalized = []
    previous_time = -1.0
    for point in points:
        if len(point.positions) != len(names) or not isfinite(point.time_from_start):
            raise ValueError("invalid trajectory point dimensions")
        if point.time_from_start < 0.0 or point.time_from_start <= previous_time:
            raise ValueError("time_from_start must be nonnegative and strictly increasing")
        positions = tuple(float(point.positions[source_index[name]]) for name in required)
        for name, value in zip(required, positions):
            lower, upper = limits[name]
            if not isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} position is outside limits")
        normalized.append(RawPoint(positions, float(point.time_from_start)))
        previous_time = point.time_from_start
    return NormalizedTrajectory(required, tuple(normalized))


def sample_trajectory(trajectory, elapsed_s):
    points = trajectory.points
    if elapsed_s <= points[0].time_from_start:
        return points[0].positions
    if elapsed_s >= points[-1].time_from_start:
        return points[-1].positions
    right = bisect_right([point.time_from_start for point in points], elapsed_s)
    left_point, right_point = points[right - 1], points[right]
    ratio = (elapsed_s - left_point.time_from_start) / (
        right_point.time_from_start - left_point.time_from_start
    )
    return tuple(
        start + ratio * (end - start)
        for start, end in zip(left_point.positions, right_point.positions)
    )


def execution_outcome(
    final_error,
    feedback_age,
    elapsed,
    duration,
    tolerance,
    stale_timeout,
    goal_time_tolerance,
):
    if feedback_age > stale_timeout:
        return "stale_feedback"
    if elapsed < duration:
        return "running"
    if final_error <= tolerance:
        return "succeeded"
    if elapsed > duration + goal_time_tolerance:
        return "goal_tolerance_violated"
    return "running"
