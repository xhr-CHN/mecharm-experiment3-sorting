import pytest

from mecharm_pick_place.trajectory import (
    RawPoint,
    execution_outcome,
    normalize_trajectory,
    sample_trajectory,
)


ARM = ("j1", "j2")
LIMITS = {"j1": (-1.0, 1.0), "j2": (-2.0, 2.0)}


def test_reorders_and_interpolates_every_point():
    trajectory = normalize_trajectory(
        ("j2", "j1"),
        (RawPoint((0.0, 0.0), 0.0), RawPoint((2.0, 1.0), 2.0)),
        ARM,
        LIMITS,
    )
    assert trajectory.points[1].positions == (1.0, 2.0)
    assert sample_trajectory(trajectory, 1.0) == pytest.approx((0.5, 1.0))


@pytest.mark.parametrize("names", [("j1",), ("j1", "j1"), ("j1", "bad")])
def test_rejects_invalid_joint_sets(names):
    with pytest.raises(ValueError):
        normalize_trajectory(
            names,
            (RawPoint(tuple(0.0 for _ in names), 0.0),),
            ARM,
            LIMITS,
        )


def test_rejects_time_and_limit_errors():
    with pytest.raises(ValueError):
        normalize_trajectory(
            ARM,
            (RawPoint((0.0, 0.0), 1.0), RawPoint((0.0, 0.0), 1.0)),
            ARM,
            LIMITS,
        )
    with pytest.raises(ValueError):
        normalize_trajectory(ARM, (RawPoint((1.1, 0.0), 0.0),), ARM, LIMITS)


def test_execution_terminal_policy():
    assert execution_outcome(0.01, 0.1, 2.0, 2.0, 0.02, 0.5, 1.0) == "succeeded"
    assert execution_outcome(0.01, 0.6, 1.0, 2.0, 0.02, 0.5, 1.0) == "stale_feedback"
    assert execution_outcome(0.1, 0.1, 3.1, 2.0, 0.02, 0.5, 1.0) == "goal_tolerance_violated"
