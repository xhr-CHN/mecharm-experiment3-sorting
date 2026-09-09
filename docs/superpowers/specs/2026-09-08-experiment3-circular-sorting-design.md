# Experiment 3 Circular Sorting Design

## Goal

Extend the existing experiment two mechArm 270 Pi Isaac Sim workflow into an automatic desktop sorting task for tennis balls and pencils. The simulation must process at least six objects from camera detections, classify at least two object types, place each type into its matching bin, return safely, and record success and exception states.

## Requirement Boundary

The attached experiment requirements define the acceptance constraints: a fixed overhead RGB camera publishes `sensor_msgs/Image`; a recognition node publishes `vision_msgs/Detection2DArray` with class, bounding box, and confidence; task control uses a state machine or BehaviorTree; the robot handles at least four to six fixed pickup grids, at least two classes, six objects with at least five correct placements, and at least two exception types; one launch entry starts the complete simulation system.

The simulation does not require arbitrary-position grasping. Pickup positions remain fixed and are selected from the detected grid ID. The user-requested layout is a ring of grid cells around the robot base, so the robot and gripper remain in the center while objects are distributed around it.

## Recommended Architecture

The implementation stays inside the existing `实验三` ROS 2 package and reuses its mechArm model, adaptive gripper, TCP/trajectory bridge, fixed-point motion, object attachment, recorder, and safety behavior.

```text
Isaac Sim scene
  overhead camera -> /camera/image_raw
  ring-layout objects and bins
        |
        v
sim_object_detector -> /mecharm/detections
        |
        v
sorting_task_node -> sorting_state_machine
        |
        v
existing fixed-point motion and gripper chain
        |
        v
placement verification + JSONL task logs
```

The detector has a stable simulation mode for the first end-to-end run and exposes the same detection message contract required by the experiment. It may later be switched to the experiment one tennis/pencil model without changing the task controller.

## Ring Layout

The scene contains six fixed pickup cells arranged around the robot base at a radius that is verified against the existing numeric IK and safe-height limits. The initial layout uses a hexagonal ring rather than a full 360-degree continuous workspace:

```text
                 G1  tennis/pencil
          G6                         G2

                  mechArm base

          G5                         G3
                 G4  tennis/pencil
```

Each cell has:

- a world-space pickup center for the existing fixed-point trajectory;
- an image-space rectangle or center for detection-to-grid association;
- an occupancy/object identifier used by the simulator;
- a reachable pre-grasp height and object-top height.

The ring radius, cell coordinates, camera pose, bin coordinates, and object class names are configuration values in `config/experiment3_sorting.yaml`. The controller may choose the next executable detection, but it must not receive a manually selected grid, class, or bin.

The six object cells and two bins occupy eight evenly spaced positions on one circle centered on the robot base. Each bin is an open-top container with a bottom and four collision walls:

- `tennis_ball` -> tennis-ball bin;
- `pencil` -> pencil bin.

## Components and Interfaces

### Scene and camera

Add the ring-layout table markers, six objects, two bins, and an overhead RGB camera to the Isaac scene builder. The camera publishes `/camera/image_raw` as `sensor_msgs/msg/Image`.

#### Pencil model

Each pencil is one dynamic rigid body rooted at its existing stable object path. Its visible geometry consists of a green cylindrical shaft, a natural-wood sharpened section, and a short black graphite point. The shaft is approximately 90 mm long with a 15 mm diameter; the wooden section is approximately 22.5 mm long and the graphite point approximately 7.5 mm long. The shaft and both tip sections all provide collision geometry under the same rigid-body root.

Only the shaft supplies collision geometry. The sharpened wood and graphite point are non-colliding visual children of the same rigid body, preventing the narrow tip from catching on the table while preserving one-piece motion and attachment. Keep the current 0.018 kg mass, object IDs, ring positions, detector class, grasp targets, and sorting behavior unchanged.

#### Tennis-ball size

Each simulated tennis ball uses a 20 mm physical radius instead of the earlier 23 mm radius. Its deterministic observation box is 40 by 40 pixels instead of 46 by 46 pixels. Keep the existing mass, color, object IDs, fixed grid coordinates, YOLO class name, grasp height, attachment distance, and sorting behavior unchanged so this remains a visual/geometry refinement rather than a motion retune.

### Simulation detector

Create `sim_object_detector.py` in the `mecharm_pick_place` package. It subscribes to `/camera/image_raw`, identifies configured tennis-ball and pencil instances, and publishes `/mecharm/detections` as `vision_msgs/msg/Detection2DArray`. Each detection includes:

- class hypothesis name `tennis_ball` or `pencil`;
- confidence score;
- 2D center and width/height;
- a stable source/object identifier when available for simulation verification.

The node rejects detections below the configured confidence threshold. Unknown objects are either published with class `unknown` for exception testing or omitted, according to the scenario configuration.

### Sorting controller

Create `sorting_task_node.py` and `sorting_state_machine.py`. The controller subscribes to `/mecharm/detections`, maps the bounding-box center to one of `G1` through `G6`, maps the class to a bin, and sends one fixed-point pick-and-place request at a time. It consumes the existing motion result/status topics and publishes:

- `/mecharm/sorting_status` (`std_msgs/String`, JSON state events);
- `/mecharm/sorting_result` (`std_msgs/String`, JSON per-object result);
- `/mecharm/sorting_start` (`std_srvs/Trigger`), optional manual start only;
- `/mecharm/sorting_stop` (`std_srvs/Trigger`).

The optional start service starts the complete autonomous batch; it does not accept target coordinates or class arguments.

### State machine

The state sequence is:

```text
IDLE -> WAIT_DETECTIONS -> SELECT_TARGET -> CHECK_TARGET
  -> HOME -> ABOVE_PICK -> DESCEND_PICK -> CLOSE_GRIPPER
  -> ATTACH_OBJECT -> LIFT -> ABOVE_BIN -> DESCEND_BIN
  -> OPEN_GRIPPER -> DETACH_OBJECT -> VERIFY_PLACE -> RETURN_HOME
  -> NEXT_TARGET or DONE
```

Failure states are explicit:

- `EMPTY_GRID`: no detection is associated with a configured cell;
- `UNKNOWN_CLASS`: detection class has no configured bin;
- `UNKNOWN_GRID`: detection center is outside all grid regions;
- `UNREACHABLE`: IK or reachability check rejects the fixed point;
- `PICK_FAILED`: motion, gripper, or attachment result fails;
- `PLACE_FAILED`: object is not in the selected bin after release;
- `TIMEOUT` or `STOP_REQUESTED`.

For a recoverable target failure, the robot publishes a hold/safe command, returns HOME when the measured state is fresh, records the failure, and continues only when the next action is safe. A stale state, collision, joint-limit violation, or unsafe return path terminates the batch in `SAFE_STOP`.

## Launch and Configuration

Create `experiment3_sorting.launch.py` to start the detector, sorting controller, existing motion/bridge nodes, recorder, and Isaac scene startup wrapper. The launch accepts `project_root`, `config`, `start_isaac`, and `scenario` arguments. The Windows Isaac process remains compatible with the current TCP 8765 bridge and the existing `start_isaac.ps1` workflow.

Create `config/experiment3_sorting.yaml` with:

- six ring cell world coordinates;
- six image-space grid regions;
- tennis-ball and pencil object definitions;
- tennis-ball and pencil bin coordinates;
- camera topic and detection topic;
- safe height, grasp height, attach distance, confidence threshold, and timeouts;
- normal six-object acceptance scenario and exception-test scenarios.

All ring coordinates must be checked against IK before the full run. The existing validated gripper behavior and safe pre-close clearance remain the baseline; arm gain/damping changes are out of scope.

## Logging and Acceptance Evidence

Write timestamped outputs under `实验三/results/experiment3/`:

- `detections.jsonl`: timestamp, object ID, class, confidence, bbox, grid;
- `task_events.jsonl`: state transitions and terminal states;
- `sorting_results.jsonl`: class, source grid, selected bin, pick result, place result, error code;
- `exception_test.md`: setup, expected behavior, observed behavior;
- `summary.yaml`: totals and acceptance booleans;
- `demo.mp4`: complete autonomous run.

The normal scenario must contain six tennis-ball/pencil objects with at least five correct placements. Separate exception runs must demonstrate at least two of empty-grid, unknown-class, unreachable, or grasp-failure handling. The acceptance checklist also verifies no visible collision, no joint-limit violation, no manual target selection after start, and successful startup through one launch command.

## Testing Strategy

Before Isaac runtime testing:

1. Unit-test ring geometry and bounding-box-center to grid mapping, including boundary and outside cases.
2. Unit-test class-to-bin mapping and unknown-class handling.
3. Unit-test state transitions for success, empty grid, unknown class, unreachable target, grasp failure, timeout, and safe stop.
4. Contract-test the launch file, configuration keys, detection message fields, and log schema.
5. Run the existing experiment two test suite without changing its dynamics baseline.

The user runs the expensive Isaac end-to-end test. A successful run should show six detection records, at least five `PLACE_SUCCESS` records, two handled exception records in the exception scenarios, continuous joint feedback, and no collision or limit error. A failure is indicated by a missing camera topic, missing detections, a target outside IK reach, stale joint feedback, attachment failure, visible collision, or fewer than five correct placements.

The motion sequence keeps the experiment-two calibrated five-smoothstep profile and stage strategy, with experiment-three speeds set to 20 degrees per second for ordinary travel and 3 degrees per second for grasp/lift/end-stage travel. It performs one HOME at batch start, releases above the bin at z=0.133 m without descending into the bin or returning HOME between targets, and performs one final HOME after the last target.

## Out of Scope

- arbitrary free-space grasping;
- a new robot model or a different gripper;
- changing the established arm gains/damping;
- real-camera calibration and Jetson deployment;
- deleting or overwriting existing experiment two results.

