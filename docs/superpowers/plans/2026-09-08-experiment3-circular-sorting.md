# Experiment 3 Circular Sorting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the local experiment two mechArm 270 Pi Isaac Sim project into an autonomous circular desktop sorter for tennis balls and pencils.

**Architecture:** Keep the existing Isaac Sim mechArm, adaptive gripper, TCP/trajectory bridge, fixed-point motion, attachment monitor, and recorder. Add a ring-layout scene, an image-to-detection node with the required `vision_msgs/Detection2DArray` contract, a pure sorting state machine, a ROS 2 sorting controller, configuration, logs, and one experiment-three launch entry. The detector is simulation-stable first and can later be replaced by the experiment-one YOLO inference node without changing task decisions.

**Tech Stack:** Isaac Sim 5.1, Python 3, ROS 2 Humble, `rclpy`, `sensor_msgs`, `vision_msgs`, `std_msgs`, `std_srvs`, existing `mecharm_pick_place` package, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-experiment3-circular-sorting-design.md`

## Global Constraints

- Work only in the local `E:\机器人集成小组项目` workspace.
- Do not run `git pull`, `git fetch`, `git push`, GitHub synchronization, or remote publishing.
- Preserve the existing experiment two arm gain/damping baseline and existing tests.
- Use the existing Elephant Robotics mechArm 270 Pi adaptive gripper.
- Place six pickup cells and two open-top bins on one eight-position circle centered on the robot base.
- Use object classes `tennis_ball` and `pencil` and map them to separate bins.
- Do not accept a manually selected grid, object class, or bin as a task input.
- Keep pickup positions fixed per grid; arbitrary-position grasping is out of scope.
- The normal scenario must target six objects and at least five correct placements.
- Demonstrate at least two exception types and persist detection, task, grasp, and placement results.

---

### Task 1: Add pure ring geometry, detection mapping, and result contracts

**Files:**
- Create: `实验三/src/mecharm_pick_place/mecharm_pick_place/sorting_types.py`
- Test: `实验三/src/mecharm_pick_place/test/test_sorting_types.py`

**Interfaces:**
- Produces `GridCell`, `DetectionRecord`, `SortingTarget`, `SortingResult`, `FailureCode`, `ring_grid_centers()`, `find_grid_for_pixel()`, and `bin_for_class()` for later tasks.

- [ ] **Step 1: Write failing tests for the six-cell ring and mappings**

```python
def test_ring_has_six_cells_around_base():
    cells = ring_grid_centers(center=(0.0, 0.0), radius=0.20)
    assert tuple(cells) == ("G1", "G2", "G3", "G4", "G5", "G6")
    assert all(abs((x * x + y * y) ** 0.5 - 0.20) < 1e-6 for x, y, _ in cells.values())

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
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run from `E:\机器人集成小组项目\实验三`:

```powershell
python -m pytest .\src\mecharm_pick_place\test\test_sorting_types.py -q
```

Expected: collection or import failure because the new sorting contracts do not exist.

- [ ] **Step 3: Implement minimal typed contracts and mapping functions**

Implement `ring_grid_centers(center, radius, z=0.05, start_angle_deg=90.0)` with six evenly spaced cells, inclusive rectangle checks in `find_grid_for_pixel()`, and exact class lookup in `bin_for_class()`. Return immutable dataclasses where practical and never infer a bin for an unknown class.

- [ ] **Step 4: Run focused tests**

```powershell
python -m pytest .\src\mecharm_pick_place\test\test_sorting_types.py -q
```

Expected: all focused tests pass.

### Task 2: Add the circular sorting state machine

**Files:**
- Create: `实验三/src/mecharm_pick_place/mecharm_pick_place/sorting_state_machine.py`
- Test: `实验三/src/mecharm_pick_place/test/test_sorting_state_machine.py`

**Interfaces:**
- Consumes `SortingTarget` and motion/attachment/placement outcomes.
- Produces deterministic states `IDLE`, `WAIT_DETECTIONS`, `SELECT_TARGET`, `CHECK_TARGET`, `HOME`, `ABOVE_PICK`, `DESCEND_PICK`, `CLOSE_GRIPPER`, `ATTACH_OBJECT`, `LIFT`, `ABOVE_BIN`, `DESCEND_BIN`, `OPEN_GRIPPER`, `DETACH_OBJECT`, `VERIFY_PLACE`, `RETURN_HOME`, `NEXT_TARGET`, `DONE`, and `SAFE_STOP`.

- [ ] **Step 1: Write failing transition tests**

```python
def test_successful_target_reaches_next_target():
    machine = SortingStateMachine()
    machine.start()
    machine.set_target(SortingTarget("obj1", "tennis_ball", "G1", (0.2, 0.0, 0.05)))
    for _ in range(machine.motion_step_count):
        machine.motion_succeeded()
    machine.gripper_succeeded()
    machine.attachment_succeeded()
    for _ in range(machine.motion_step_count * 3):
        machine.motion_succeeded()
    machine.gripper_succeeded()
    machine.placement_succeeded()
    assert machine.state in {SortState.NEXT_TARGET, SortState.DONE}

def test_unknown_class_is_recoverable():
    machine = SortingStateMachine()
    machine.start()
    machine.set_target(SortingTarget("obj2", "unknown", "G2", (0.2, 0.0, 0.05)))
    assert machine.state is SortState.SAFE_STOP or machine.failure_code is FailureCode.UNKNOWN_CLASS

def test_motion_failure_enters_safe_stop():
    machine = SortingStateMachine()
    machine.start()
    machine.motion_failed(FailureCode.UNREACHABLE, "fixed point rejected")
    assert machine.state is SortState.SAFE_STOP
```

- [ ] **Step 2: Run tests and verify the expected failure**

```powershell
python -m pytest .\src\mecharm_pick_place\test\test_sorting_state_machine.py -q
```

Expected: import or missing-method failures.

- [ ] **Step 3: Implement the transition table and failure policy**

Keep transitions pure and explicit. `UNKNOWN_CLASS` and `UNKNOWN_GRID` are logged and skipped before motion. `UNREACHABLE`, stale feedback, collision, joint-limit violation, and unsafe attachment enter `SAFE_STOP`. Never retry motion while the previous action is non-terminal.

- [ ] **Step 4: Run focused and existing state-machine tests**

```powershell
python -m pytest .\src\mecharm_pick_place\test\test_sorting_types.py .\src\mecharm_pick_place\test\test_sorting_state_machine.py .\src\mecharm_pick_place\test\test_state_machine.py -q
```

Expected: all tests pass.

### Task 3: Add the simulation detector node and ROS message contract

**Files:**
- Create: `实验三/src/mecharm_pick_place/mecharm_pick_place/sim_object_detector.py`
- Modify: `实验三/src/mecharm_pick_place/setup.py`
- Modify: `实验三/src/mecharm_pick_place/package.xml`
- Test: `实验三/src/mecharm_pick_place/test/test_sim_object_detector.py`

**Interfaces:**
- Consumes `/camera/image_raw` as `sensor_msgs/msg/Image`.
- Produces `/mecharm/detections` as `vision_msgs/msg/Detection2DArray`.
- Parameters: `image_topic`, `detection_topic`, `confidence_threshold`, `mode`, `grid_regions`, and configured simulation objects.

- [ ] **Step 1: Write tests for detection conversion and filtering**

```python
def test_detection_record_becomes_detection2d():
    message = to_detection_message(
        DetectionRecord("obj1", "tennis_ball", 0.94, 120.0, 80.0, 30.0, 30.0)
    )
    assert message.results[0].hypothesis.class_id == "tennis_ball"
    assert message.results[0].hypothesis.score == 0.94
    assert message.bbox.center.position.x == 120.0

def test_low_confidence_detection_is_filtered():
    records = filter_detections(
        [DetectionRecord("obj1", "pencil", 0.3, 1, 1, 2, 2)], threshold=0.5
    )
    assert records == []
```

- [ ] **Step 2: Run the detector tests and verify failure**

```powershell
python -m pytest .\src\mecharm_pick_place\test\test_sim_object_detector.py -q
```

Expected: missing module/function failure.

- [ ] **Step 3: Implement the simulation-stable detector**

Implement the detector as a ROS 2 node with a small pure conversion layer. Use configured ring object projections for the first simulation scenario, publish valid `Detection2DArray` messages, preserve class and confidence, and support an `unknown` scenario record for exception testing. Keep the conversion layer independent from Isaac so it remains unit-testable without ROS runtime.

- [ ] **Step 4: Add package dependencies and console entry point**

Add `vision_msgs` and `sensor_msgs` dependencies where the package manifest requires them, and add:

```text
sim_object_detector = mecharm_pick_place.sim_object_detector:main
```

- [ ] **Step 5: Run detector and package tests**

```powershell
python -m pytest .\src\mecharm_pick_place\test\test_sim_object_detector.py -q
```

Expected: focused conversion/filter tests pass. ROS import failures are recorded separately if the Windows host does not have the ROS runtime; the container build check in Task 6 is authoritative for ROS imports.

### Task 4: Build the ring-layout Isaac scene and camera configuration

**Files:**
- Modify: `实验三/simulation/isaac/start_simulation.py`
- Create: `实验三/simulation/isaac/experiment3_scene.py`
- Create: `实验三/config/experiment3_sorting.yaml`
- Test: `实验三/tests/test_experiment3_config.py`

**Interfaces:**
- Produces `/World/experiment3/work_table`, six ring cells, six tennis-ball/pencil objects, two bins, and `/World/top_camera`.
- Keeps `/World/mecharm_270_pi`, the existing adaptive-gripper baseline, and the TCP 8765 control path unchanged.

- [ ] **Step 1: Write configuration contract tests**

```python
def test_configuration_has_six_ring_cells_and_two_classes():
    config = load_yaml(Path("config/experiment3_sorting.yaml"))
    assert list(config["grid_centers"]) == ["G1", "G2", "G3", "G4", "G5", "G6"]
    assert set(config["class_bins"]) == {"tennis_ball", "pencil"}
    assert len(config["objects"]) == 6

def test_ring_coordinates_are_inside_table_and_have_nonzero_radius():
    config = load_yaml(Path("config/experiment3_sorting.yaml"))
    points = list(config["grid_centers"].values())
    assert all(len(point) == 3 for point in points)
    assert max(abs(point[0]) for point in points) < 0.35
    assert max(abs(point[1]) for point in points) < 0.25
```

- [ ] **Step 2: Implement YAML with six ring positions and six objects**

Use `tennis_ball` and `pencil` as the only normal classes. Place six objects and two bins at eight unique angles on the same reachable circle centered on the robot base. Store image regions separately from world positions.

- [ ] **Step 3: Implement scene fixtures in a focused helper**

Create the centered table, object prims, open-top bins (bottom plus four collision walls), and camera in `experiment3_scene.py`. Keep existing URDF import, gripper drive, selective collision filtering, and baseline dynamics in `start_simulation.py`; call the helper after the robot exists. Use stable prim paths `/World/experiment3/objects/<object_id>` so attachment and placement verification can identify the selected object.

- [ ] **Step 4: Add camera publishing without changing the existing transport baseline**

Create the overhead camera and configure the Isaac ROS camera bridge to publish `/camera/image_raw` when the ROS 2 transport is selected. When the current TCP transport is used, expose the same camera contract through the local ROS process or a documented camera adapter; do not reintroduce cross-host DDS discovery into the established TCP joint bridge.

- [ ] **Step 5: Run configuration tests**

```powershell
python -m pytest .\tests\test_experiment3_config.py -q
```

Expected: all static configuration tests pass before starting Isaac.

### Task 5: Implement the sorting controller and multi-object attachment flow

**Files:**
- Create: `实验三/src/mecharm_pick_place/mecharm_pick_place/sorting_task_node.py`
- Modify: `实验三/simulation/isaac/grasp_monitor.py`
- Modify: `实验三/config/experiment3_sorting.yaml`
- Test: `实验三/src/mecharm_pick_place/test/test_sorting_task_node.py`

**Interfaces:**
- Consumes `/mecharm/detections`, `/mecharm/motion_result`, and `/joint_states` feedback.
- Publishes `/mecharm/sorting_status`, `/mecharm/sorting_result`, `/mecharm/joint_target`, `/mecharm/gripper_command`.
- Services `/mecharm/sorting_start` and `/mecharm/sorting_stop` accept no target arguments.

- [ ] **Step 1: Write controller tests for automatic selection and exceptions**

```python
def test_controller_selects_class_bin_from_detection():
    controller = SortingControllerCore(config=fixture_config())
    controller.receive_detections([detection("obj1", "tennis_ball", "G2")])
    target = controller.select_next_target()
    assert target.grid_id == "G2"
    assert target.bin_position == fixture_config()["class_bins"]["tennis_ball"]

def test_controller_skips_unknown_class_without_motion_command():
    controller = SortingControllerCore(config=fixture_config())
    controller.receive_detections([detection("obj2", "unknown", "G3")])
    controller.select_next_target()
    assert controller.motion_commands == []
    assert controller.results[-1].error_code == "UNKNOWN_CLASS"

def test_controller_records_empty_grid():
    controller = SortingControllerCore(config=fixture_config())
    controller.receive_detections([])
    controller.finish_scan()
    assert controller.results[-1].error_code == "EMPTY_GRID"
```

- [ ] **Step 2: Run focused tests and verify failure**

```powershell
python -m pytest .\src\mecharm_pick_place\test\test_sorting_task_node.py -q
```

Expected: missing controller core failure.

- [ ] **Step 3: Implement pure controller core**

Convert detections to targets by bbox center and configured regions, sort candidates deterministically by confidence then grid ID, reject duplicate occupied grids, map class to bin, and emit structured result events. Keep ROS publisher/subscriber code thin so the selection and failure policy are testable without a running graph.

- [ ] **Step 4: Connect the core to the existing fixed-point motion chain**

For each target, publish the existing HOME/pick/place joint or pose commands and gripper commands through the established topics. Use the target grid world coordinate for pick and the class bin coordinate for place. Only mark attachment after the measured gripper/object distance meets the configured threshold. Add selected object prim path support to `GraspMonitor` while preserving current single-object behavior for experiment two.

- [ ] **Step 5: Add placement verification and JSONL logs**

Verify the selected object is within the configured bin tolerance after detach. Record one JSON object per detection, state transition, and object result under `results/experiment3/`. Use atomic append/flush behavior and include timestamp, object ID, class, grid, bin, success, and error code.

- [ ] **Step 6: Run controller tests and all existing non-runtime tests**

```powershell
python -m pytest .\src\mecharm_pick_place\test .\tests -q
```

Expected: all pure Python and contract tests pass; no Isaac process is required for this checkpoint.

### Task 6: Add one Launch entry and exception scenarios

**Files:**
- Create: `实验三/src/mecharm_pick_place/launch/experiment3_sorting.launch.py`
- Modify: `实验三/src/mecharm_pick_place/setup.py`
- Create: `实验三/config/experiment3_exception_empty_unknown.yaml`
- Create: `实验三/config/experiment3_exception_unreachable.yaml`
- Modify: `实验三/docs/testing.md`
- Test: `实验三/tests/test_experiment3_launch_contract.py`

**Interfaces:**
- Launch command: `ros2 launch mecharm_pick_place experiment3_sorting.launch.py project_root:=/workspace/mecharm_exp3 config:=... scenario:=normal`
- Starts the detector, sorting controller, existing bridges/recorder, and the Isaac startup wrapper using the existing Windows/TCP topology.

- [ ] **Step 1: Write launch contract tests**

```python
def test_experiment3_launch_mentions_required_nodes_and_config():
    text = Path("src/mecharm_pick_place/launch/experiment3_sorting.launch.py").read_text()
    for token in ("sim_object_detector", "sorting_task_node", "pick_place_recorder_node", "experiment3_sorting.yaml"):
        assert token in text

def test_exception_configs_cover_empty_unknown_and_unreachable():
    assert Path("config/experiment3_exception_empty_unknown.yaml").is_file()
    assert Path("config/experiment3_exception_unreachable.yaml").is_file()
```

- [ ] **Step 2: Run launch tests and verify failure**

```powershell
python -m pytest .\tests\test_experiment3_launch_contract.py -q
```

Expected: missing launch/config failure.

- [ ] **Step 3: Implement the launch description**

Declare `project_root`, `config`, `result_root`, `scenario`, `start_isaac`, and `windows_root`. Start the required ROS nodes with the selected YAML and preserve the current `start_isaac.ps1` invocation pattern. Do not add any Git or remote command to launch or helper scripts.

- [ ] **Step 4: Add exception scenario configurations**

The first exception scenario contains an empty grid and an unknown object. The second places one target outside the configured reachable workspace. The controller must skip or safe-stop as specified and write the corresponding error codes.

- [ ] **Step 5: Document the two-terminal fallback and single-launch path**

Document the preferred single command, the existing container build command, the Isaac Windows prerequisites, expected topics, expected logs, and the safe shutdown order. State clearly that GitHub synchronization is not part of the workflow.

- [ ] **Step 6: Run launch contract tests**

```powershell
python -m pytest .\tests\test_experiment3_launch_contract.py -q
```

Expected: all launch contract tests pass.

### Task 7: Build the ROS workspace and perform user-run Isaac acceptance

**Files:**
- Modify: `实验三/README.md`
- Create: `实验三/results/experiment3/README.md`
- Create: `实验三/results/experiment3/.gitkeep`

- [ ] **Step 1: Run local static and unit validation**

```powershell
Set-Location 'E:\机器人集成小组项目\实验三'
python -m pytest .\src\mecharm_pick_place\test .\tests -q
```

Expected: all tests pass. If ROS packages are unavailable on Windows, run the same tests in the project container after the build step.

- [ ] **Step 2: Build the existing local ROS container workspace**

```powershell
docker compose -f '.\docker-compose.yml' exec moveit bash -lc "source /opt/ros/humble/setup.bash && colcon --log-base /opt/mecharm_ws/log build --base-paths /workspace/mecharm_exp3/simulation/urdf/mycobot_description /workspace/mecharm_exp3/src --build-base /opt/mecharm_ws/build --install-base /opt/mecharm_ws/install --symlink-install"
```

Expected: the package builds and installs `experiment3_sorting.launch.py`, `sim_object_detector`, and `sorting_task_node` without traceback.

- [ ] **Step 3: Rebuild the Isaac scene in Windows**

```powershell
& '.\scripts\start_isaac.ps1' -ProjectRoot 'E:\机器人集成小组项目\实验三' -RebuildScene -BuildSceneOnly
```

Expected: the USD contains the mechArm, six circular pickup cells, six objects, two bins, and the overhead camera. Stop and fix any IK or scene-build error before running the batch.

- [ ] **Step 4: Run the normal six-object acceptance personally**

Use the new launch command from the container after the scene is loaded. Expected evidence:

```text
/camera/image_raw exists
/mecharm/detections exists
six object detections are logged
at least five PLACE_SUCCESS results
no collision or joint-limit error
robot returns HOME
results/experiment3/summary.yaml exists
```

- [ ] **Step 5: Run the two exception scenarios personally**

Run the empty/unknown scenario and the unreachable scenario separately. Expected evidence includes `EMPTY_GRID`, `UNKNOWN_CLASS`, and `UNREACHABLE` or `SAFE_STOP`, with no unsafe motion after the failure.

- [ ] **Step 6: Update the local acceptance summary only**

Record actual counts, videos, logs, and observed failures in `results/experiment3/README.md` and the experiment report. Do not commit, push, pull, fetch, or synchronize with GitHub.

---

## Plan Self-Review

- The attached requirements are covered by Tasks 3, 4, 5, 6, and 7: image topic, detection message, state-machine control, fixed grids, two classes, six-object count, five correct placements, exceptions, logs, and one launch entry.
- The user-requested ring layout is explicit in the global constraints, scene task, configuration task, and acceptance task.
- No task depends on an undefined later function: `sorting_types.py` defines the target/result contracts used by the state machine and controller; the launch task names the entry points added by the detector/controller task.
- The plan contains no unresolved placeholders or remote synchronization steps.
- Existing experiment two dynamics and tests are preserved; the expensive Isaac run is left to the user after lightweight checks.

