from pathlib import Path

import yaml


ROOT = Path("src/mecharm_pick_place")
SOURCE = (ROOT / "mecharm_pick_place/cartesian_direct_pick_place.py").read_text(
    encoding="utf-8"
)


def test_cartesian_direct_mode_uses_collision_free_ik_and_vertical_tool():
    config = yaml.safe_load(
        (ROOT / "config/direct_cartesian_pick_place.yaml").read_text(encoding="utf-8")
    )["cartesian_direct_pick_place"]["ros__parameters"]
    assert config["pick_xyz"] == [0.18, 0.09, 0.025]
    assert config["place_xyz"] == [0.18, -0.08, 0.025]
    assert config["position_tolerance"] == 0.008
    assert config["vertical_axis_dot_min"] == 0.975
    assert config["pregrasp_clearance"] == 0.10
    assert config["pick_close_clearance"] == 0.04
    assert "grasp_clearance" not in config
    assert "UrdfNumericIK" in SOURCE
    assert config["gripper_open"] == 0.15
    assert config["gripper_closed"] == -0.75
    assert config["gripper_speed_rad_s"] == 0.12
    assert config["gripper_wait_before_sec"] == 1.0
    assert config["gripper_wait_after_sec"] == 1.5
    assert "NUMERIC_IK_SUCCESS" in SOURCE
    assert "CARTESIAN_DIRECT_PICK_PLACE_SUCCESS" in SOURCE


def test_gripper_closes_before_approach_then_opens_descends_and_closes():
    stages = (
        "INITIAL_GRIPPER_OPEN",
        "PICK_PREGRASP",
        "PICK_GRASP",
        "GRIPPER_CLOSE_ON_OBJECT",
    )
    offsets = [SOURCE.index(stage) for stage in stages]
    assert offsets == sorted(offsets)
    assert "GRIPPER_WAIT_BEFORE" in SOURCE
    assert "GRIPPER_WAIT_AFTER" in SOURCE
    assert "GRIPPER_OPEN_ABOVE_OBJECT" not in SOURCE
    assert 'node.solve("PLACE_GRASP", place, pick_close, command, grasp_tool_x)' in SOURCE
    assert 'grasp_yaw_offset_deg' in SOURCE
