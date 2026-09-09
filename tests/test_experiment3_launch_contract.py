from pathlib import Path


ROOT = Path(__file__).parents[1]
LAUNCH = ROOT / "src" / "mecharm_pick_place" / "launch" / "experiment3_sorting.launch.py"


def test_experiment3_launch_mentions_required_nodes_and_config():
    text = LAUNCH.read_text(encoding="utf-8")
    for token in (
        "virtual_camera_node",
        "yolo_classifier_node",
        "vision_request",
        "sorting_task_node",
        "pick_place_recorder_node",
        "experiment3_sorting.yaml",
        "Experiment3",
    ):
        assert token in text


def test_exception_configs_cover_empty_unknown_and_unreachable():
    assert (ROOT / "config" / "experiment3_exception_empty_unknown.yaml").is_file()
    assert (ROOT / "config" / "experiment3_exception_unreachable.yaml").is_file()


def test_real_camera_and_model_contract_are_present():
    launch_text = LAUNCH.read_text(encoding="utf-8")
    config_text = (ROOT / "config" / "experiment3_sorting.yaml").read_text(encoding="utf-8")
    assert "virtual_camera_node" in launch_text
    assert "port\": 8766" in launch_text
    assert "yolo_classifier_node" in launch_text
    assert "pencil_tennis_yolo26n_best.pt" in config_text
    assert (ROOT / "models" / "pencil_tennis_yolo26n_best.pt").is_file()


def test_joint_tcp_bridge_has_heartbeat_recovery():
    ros_bridge = (
        ROOT / "src" / "mecharm_pick_place" / "mecharm_pick_place" / "tcp_bridge_node.py"
    ).read_text(encoding="utf-8")
    isaac_bridge = (ROOT / "simulation" / "isaac" / "tcp_joint_bridge.py").read_text(
        encoding="utf-8"
    )
    assert 'b\'{"type":"heartbeat"}\\n\'' in ros_bridge
    assert "last_message_time" in isaac_bridge
    assert "> 3.0" in isaac_bridge


def test_sorting_uses_high_drop_and_skips_inter_object_home():
    source = (
        ROOT / "src" / "mecharm_pick_place" / "mecharm_pick_place" / "sorting_task_node.py"
    ).read_text(encoding="utf-8")
    config = (ROOT / "config" / "experiment3_sorting.yaml").read_text(encoding="utf-8")
    assert 'carry_clearance = float(self.core.config.get("carry_clearance", 0.108))' in source
    assert '"GRIPPER_OPEN_ABOVE_BIN"' in source
    assert '"PLACE_GRASP"' not in source
    assert '"RETREAT"' not in source
    assert '"FINAL_HOME"' in source
    assert '"RETURN_HOME"' in source
    assert "carry_clearance: 0.108" in config


def test_isaac_gripper_dynamics_match_experiment_two_baseline():
    source = (ROOT / "simulation" / "isaac" / "start_simulation.py").read_text(
        encoding="utf-8"
    )
    assert "set_default_drive_strength(1000)" in source
    assert "set_default_position_drive_damping(30000)" in source
    assert "GetStiffnessAttr().Set(150.0)" in source
    assert "GetDampingAttr().Set(60.0)" in source
    assert "GetMaxForceAttr().Set(20.0)" in source
    assert '"gripper_base_to_gripper_left2": (-42.972, 8.594)' in source
    assert '"gripper_base_to_gripper_right3": (-8.594, 42.972)' in source
    assert '("CLOSE", -0.75, 7.0)' in source
    assert "physical_tip_pad" not in source


def test_sorting_motion_keeps_joint_feedback_running_during_motion():
    probe = (ROOT / "src" / "mecharm_pick_place" / "mecharm_pick_place" / "direct_motion_probe.py").read_text(
        encoding="utf-8"
    )
    task = (ROOT / "src" / "mecharm_pick_place" / "mecharm_pick_place" / "sorting_task_node.py").read_text(
        encoding="utf-8"
    )
    assert "ReentrantCallbackGroup" in probe
    assert "MultiThreadedExecutor" in task
    assert "external_executor_spins" in probe
    assert "self.external_executor_spins = True" in task


def test_sorting_uses_one_batch_vision_request_before_fixed_grid_motion():
    source = (
        ROOT / "src" / "mecharm_pick_place" / "mecharm_pick_place" / "sorting_task_node.py"
    ).read_text(encoding="utf-8")
    yolo = (
        ROOT / "src" / "mecharm_pick_place" / "mecharm_pick_place" / "yolo_classifier_node.py"
    ).read_text(encoding="utf-8")
    config = (ROOT / "config" / "experiment3_sorting.yaml").read_text(encoding="utf-8")
    assert 'String(data="ALL")' in source
    assert "_waiting_batch" in source
    assert "batch_classes_by_grid" in source
    assert 'grid_id.upper() == "ALL"' in yolo
    assert "batch_frame_count: 6" in config
    assert "batch_minimum_votes: 4" in config
