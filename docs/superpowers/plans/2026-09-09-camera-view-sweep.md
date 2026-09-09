# Camera View Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Render, score, document, and apply the best experiment-three overhead camera view.

**Architecture:** A standalone Isaac script renders 13 deterministic poses into a timestamped result folder. A container-side evaluator runs the experiment-one YOLO model, creates annotated evidence, aggregates five-frame stability, and writes the winning pose for production application.

**Tech Stack:** Isaac Sim Replicator, OpenCV, Ultralytics YOLO, CSV/JSON/YAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-camera-view-sweep-design.md`

## Constraints

- Do not change object or robot coordinates during the sweep.
- Use 13 views and five frames per view.
- Keep every source frame and model prediction.
- Do not synchronize GitHub.

### Task 1: Render candidate views

- Create `simulation/isaac/camera_view_sweep.py`.
- Create 13 camera poses, render five frames each, and write `view_manifest.json` plus PNG images.
- Verify the manifest has 65 frame records.

### Task 2: Evaluate with YOLO

- Create `tools/evaluate_camera_views.py`.
- Run the copied experiment-one model on every image.
- Save annotations, detections, aggregate CSV, hash, YAML, and Markdown report.
- Verify every candidate has five scored frames and one recommendation is selected.

### Task 3: Apply winner

- Read `recommended_camera.yaml`.
- Update `simulation/isaac/experiment3_scene.py` only after evidence exists.
- Re-run static tests and leave the full report in the result folder.
