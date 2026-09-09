# Fixed-Grid YOLO Camera Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed the real Isaac Sim overhead camera into the experiment-three YOLO pipeline while keeping all six pickup coordinates fixed and known.

**Architecture:** Isaac renders `/World/experiment3/top_camera` and streams JPEG frames over the existing host TCP topology. A ROS 2 image adapter publishes `/camera/image_raw`; a YOLO node receives a requested grid ID, classifies only that fixed grid ROI with the experiment-one model, and publishes the class. The sorting controller uses the requested grid's configured world coordinate and never uses a YOLO box center for grasping.

**Tech Stack:** Isaac Sim 5.1 Replicator RGB annotator, Python sockets/OpenCV, ROS 2 Humble, `sensor_msgs`, `vision_msgs`, Ultralytics YOLO26n.

**Spec:** `docs/superpowers/specs/2026-09-08-experiment3-circular-sorting-design.md`

## Global Constraints

- The six grid positions and processing order are fixed in `config/experiment3_sorting.yaml`.
- YOLO determines only `tennis_ball` or `pencil`; its bounding-box center is ignored by task control.
- The controller uses `grid_centers[grid_id]` for every grasp.
- Copy, do not move, `实验一/models/pencil_tennis_yolo26n_best.pt` into `实验三/models/`.
- Preserve experiment two and experiment three TCP joint control and gripper behavior.
- Do not run GitHub synchronization during implementation.

---

### Task 1: Add real camera stream and image adapter

**Files:**
- Create: `实验三/simulation/isaac/experiment3_camera_stream.py`
- Modify: `实验三/simulation/isaac/start_simulation.py`
- Create: `实验三/src/mecharm_pick_place/mecharm_pick_place/virtual_camera_node.py`
- Modify: `实验三/src/mecharm_pick_place/setup.py`
- Modify: `实验三/src/mecharm_pick_place/package.xml`
- Modify: `实验三/docker/Dockerfile`

Implement a length-prefixed JPEG TCP stream on port `8766`, publish decoded RGB frames as `sensor_msgs/Image`, and add the Isaac camera stream to the experiment-three control loop only.

### Task 2: Add fixed-ROI YOLO classifier

**Files:**
- Create: `实验三/src/mecharm_pick_place/mecharm_pick_place/yolo_classifier_node.py`
- Modify: `实验三/src/mecharm_pick_place/launch/experiment3_sorting.launch.py`
- Modify: `实验三/config/experiment3_sorting.yaml`
- Create: `实验三/tests/test_yolo_classifier_contract.py`

Load the copied experiment-one `.pt` model, subscribe to the image topic, accept `/mecharm/vision_request` grid IDs, crop only the configured fixed ROI, and publish one class result on `/mecharm/detections`. Publish model boxes for observation only; never make them task coordinates.

### Task 3: Change task control to sequential fixed-grid classification

**Files:**
- Modify: `实验三/src/mecharm_pick_place/mecharm_pick_place/sorting_task_node.py`
- Modify: `实验三/src/mecharm_pick_place/mecharm_pick_place/sorting_types.py`
- Modify: `实验三/src/mecharm_pick_place/test/test_sorting_task_node.py`

Add a fixed-grid target constructor and make the controller request and await classification in `G1` through `G6` order. Build every `SortingTarget.pick_position` exclusively from `grid_centers`; reject missing/unknown classes without motion.

### Task 4: Validate and document

**Files:**
- Modify: `实验三/README.md`
- Modify: `实验三/docs/testing.md`

Run pure tests and syntax checks. Document model path, TCP camera port, topics, expected logs, and the fact that YOLO coordinates are not used for grasping. Full Isaac/YOLO runtime remains user-run.

## Success Criteria

- Static tests pass.
- The Isaac scene contains `/World/experiment3/top_camera` and streams frames on TCP 8766.
- ROS exposes `/camera/image_raw` and YOLO exposes `/mecharm/detections`.
- A vision request for `G3` yields only a class decision for `G3`.
- The controller's pick target for `G3` equals the configured `grid_centers[G3]` regardless of the YOLO bbox center.
