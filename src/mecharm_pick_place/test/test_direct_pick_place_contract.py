from pathlib import Path


SOURCE = Path(
    "src/mecharm_pick_place/mecharm_pick_place/direct_pick_place_demo.py"
).read_text(encoding="utf-8")
LAUNCH = Path(
    "src/mecharm_pick_place/launch/direct_pick_place.launch.py"
).read_text(encoding="utf-8")


def test_direct_sequence_contains_calibrated_pick_place_stages():
    for token in (
        "PICK_SAFE",
        "PICK_GRASP",
        "PLACE_SAFE",
        "PLACE_GRASP",
        "GRIPPER_CLOSE",
        "DIRECT_PICK_PLACE_SUCCESS",
    ):
        assert token in SOURCE
    assert "moveit" not in SOURCE.lower()


def test_direct_arm_uses_experiment_two_fast_and_slow_speed_profile():
    assert "MAX_ARM_SPEED_RAD_S = math.radians(20.0)" in SOURCE
    assert "VERTICAL_ARM_SPEED_RAD_S = math.radians(3.0)" in SOURCE
    assert '"PICK_GRASP", "LIFT", "PLACE_GRASP", "RETREAT"' in SOURCE


def test_direct_launch_starts_only_tcp_bridge_and_direct_demo():
    assert 'executable="isaac_tcp_bridge"' in LAUNCH
    assert 'executable="direct_pick_place_demo"' in LAUNCH
    assert "move_group" not in LAUNCH
    assert "isaac_trajectory_controller" not in LAUNCH
