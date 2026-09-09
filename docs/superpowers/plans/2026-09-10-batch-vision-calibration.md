# Batch Vision Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Classify all six fixed grid objects from six consecutive frames at startup, lock the voted classes, and then execute the existing G1–G6 pick-and-place sequence without per-object vision requests.

**Architecture:** Keep the existing `/mecharm/vision_request` and `/mecharm/detections` topics. The sorting node sends one `ALL` request; the YOLO node keeps the newest six image messages, runs each configured fixed ROI six times, applies a 4-of-6 majority vote, and publishes one `Detection2DArray` containing G1–G6. The sorting node stores that batch and constructs each target from the fixed `grid_centers`, so YOLO geometry remains irrelevant to motion.

**Tech Stack:** ROS 2 Humble, Python, `rclpy`, `vision_msgs`, OpenCV, cv_bridge, Ultralytics YOLO, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-batch-vision-calibration-design.md`

## Global Constraints

- Use six frames per grid and require at least four identical class votes.
- Keep the current model, confidence threshold, camera pose, robot trajectory, speeds, gripper parameters, and physical scene unchanged.
- Use `grid_regions` only for image crops and `grid_centers` only for pickup coordinates.
- Do not add a second vision topic or change the launch command contract.
- A failed batch classification must publish `unknown` for that grid and prevent blind pickup.
- Do not include GitHub operations in the runtime launch path.

---

### Task 1: Add pure six-frame vote and batch-record helpers

**Files:**
- Modify: `src/mecharm_pick_place/mecharm_pick_place/yolo_classifier_node.py`
- Test: `src/mecharm_pick_place/test/test_yolo_classifier_node.py`

**Interfaces:**
- `majority_class(votes: Sequence[dict], minimum_votes: int) -> dict | None` returns the highest-vote class and averaged confidence, or `None` when no class reaches the minimum.
- `classify_grid_frames(model, frames: Sequence, region: Sequence[float], confidence_threshold: float, image_size: int, device: str) -> dict | None` returns the voted class, confidence, and six per-frame decisions.

- [ ] **Step 1: Write failing tests** for a 4-of-6 vote, a 3-3 tie, and six unknown frames.
- [ ] **Step 2: Run** `pytest src/mecharm_pick_place/test/test_yolo_classifier_node.py -q`; expect the helper import or behavior to fail.
- [ ] **Step 3: Implement** the helpers using `crop_fixed_roi`, `best_yolo_detection`, and the existing YOLO call. Do not use detection box centers in the returned grid decision.
- [ ] **Step 4: Run** the focused test and require all cases to pass.
- [ ] **Step 5: Commit** with `git add src/mecharm_pick_place/mecharm_pick_place/yolo_classifier_node.py src/mecharm_pick_place/test/test_yolo_classifier_node.py && git commit -m "Add six-frame fixed-ROI class voting"`.

### Task 2: Make the YOLO node process one full batch

**Files:**
- Modify: `src/mecharm_pick_place/mecharm_pick_place/yolo_classifier_node.py`
- Modify: `config/experiment3_sorting.yaml`
- Test: `src/mecharm_pick_place/test/test_yolo_classifier_node.py`

**Interfaces:**
- New parameters: `batch_frame_count: int = 6` and `batch_minimum_votes: int = 4`.
- Request payload `ALL` triggers processing in `fixed_grid_order` and publishes one `Detection2DArray` with six `Detection2D` entries.
- Each detection ID is its grid ID and its first hypothesis is the voted class or `unknown`.

- [ ] **Step 1: Add failing tests** asserting an `ALL` request requires six buffered frames and produces six grid decisions.
- [ ] **Step 2: Run** the focused test; expect failure because the node currently stores one image and accepts only one grid ID.
- [ ] **Step 3: Add** a bounded deque of six image/header pairs in `_on_image`, read the two batch parameters, and implement `_on_batch_request` that processes every configured grid over the same six frames.
- [ ] **Step 4: Publish** one array after all six grids are processed, log one `YOLO BATCH` summary, and publish `unknown` entries for low-vote grids.
- [ ] **Step 5: Run** focused tests and the existing YOLO contract tests.
- [ ] **Step 6: Commit** with `git add config/experiment3_sorting.yaml src/mecharm_pick_place/mecharm_pick_place/yolo_classifier_node.py src/mecharm_pick_place/test/test_yolo_classifier_node.py && git commit -m "Classify all fixed grids in one vision batch"`.

### Task 3: Change sorting orchestration to wait for and lock one batch

**Files:**
- Modify: `src/mecharm_pick_place/mecharm_pick_place/sorting_task_node.py`
- Test: `src/mecharm_pick_place/test/test_sorting_task_node.py`
- Test: `tests/test_experiment3_launch_contract.py`

**Interfaces:**
- `_on_detections` accepts a batch array when waiting for `ALL`, indexes records by grid ID, and stores `dict[str, tuple[str, float]]` in `_classified_by_grid`.
- `_batch_tick` sends exactly one `String(data="ALL")`, waits for the complete six-grid response, and then calls `target_for_fixed_grid` in `fixed_grid_order` without publishing another vision request.

- [ ] **Step 1: Add failing tests** for one request only, six locked classes, fixed-center target construction, and no second request during G1–G6.
- [ ] **Step 2: Run** the focused sorting tests; expect failure because the current node waits on `_waiting_grid` and publishes G1, G2, etc. separately.
- [ ] **Step 3: Replace** `_waiting_grid` flow with `_waiting_batch`, initialize `_classified_by_grid`, and handle timeout/unknown grids before motion starts.
- [ ] **Step 4: Keep** `_execute_target` unchanged so all motion, lift, release, and final HOME behavior stays identical.
- [ ] **Step 5: Run** sorting tests and launch contract tests.
- [ ] **Step 6: Commit** with `git add src/mecharm_pick_place/mecharm_pick_place/sorting_task_node.py src/mecharm_pick_place/test/test_sorting_task_node.py tests/test_experiment3_launch_contract.py && git commit -m "Run sorting from one locked vision batch"`.

### Task 4: Update runtime documentation and verify the complete package

**Files:**
- Modify: `README.md`
- Modify: `docs/testing.md`
- Test: `tests/`
- Test: `src/mecharm_pick_place/test/`

- [ ] **Step 1: Update** startup documentation to state that six-frame batch calibration occurs once before G1–G6 motion.
- [ ] **Step 2: Run** `python -m pytest tests src/mecharm_pick_place/test -q` with `PYTHONPATH` set to the package source; require all tests to pass.
- [ ] **Step 3: Run** `python -m py_compile` on the modified Python files.
- [ ] **Step 4: Build** the ROS package in the existing Docker workspace and confirm the launch package installs.
- [ ] **Step 5: Commit** with `git add README.md docs/testing.md && git commit -m "Document one-shot vision calibration workflow"`.
- [ ] **Step 6: Runtime acceptance:** with Isaac running, verify one batch request, six per-grid six-frame vote logs, no later `VISION_REQUEST`, G1–G6 results, and final HOME. Do not push unless separately requested.
