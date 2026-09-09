from pathlib import Path

import numpy as np

from mecharm_pick_place.numeric_ik import UrdfNumericIK


URDF = Path(
    "simulation/urdf/mycobot_description/urdf/mecharm_270_pi/"
    "mecharm_270_pi_adaptive_gripper.urdf"
)


def test_numeric_ik_reaches_vertical_pregrasp_without_joint6_only_solution():
    solver = UrdfNumericIK(URDF)
    solution, position_error, axis_dot = solver.solve(
        (0.18, 0.08, 0.105), np.zeros(6)
    )
    assert position_error <= 0.005
    assert axis_dot >= 0.98
    assert np.linalg.norm(np.asarray(solution[:5])) > 0.1


def test_numeric_ik_solves_complete_pick_place_without_wrist_wrap():
    solver = UrdfNumericIK(URDF)
    positions = np.zeros(6)
    for target in (
        (0.18, 0.08, 0.105),
        (0.18, 0.08, 0.025),
        (0.18, 0.08, 0.105),
        (0.18, -0.08, 0.105),
        (0.18, -0.08, 0.025),
        (0.18, -0.08, 0.105),
    ):
        previous = positions
        positions, position_error, axis_dot = solver.solve(target, positions)
        positions = np.asarray(positions)
        assert position_error <= 0.005
        assert axis_dot >= 0.98
        assert abs(positions[5] - previous[5]) < np.pi
