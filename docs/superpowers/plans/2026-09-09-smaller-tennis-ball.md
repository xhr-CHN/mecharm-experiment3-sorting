# Smaller Tennis Ball Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the experiment-three tennis-ball radius from 23 mm to 20 mm and keep simulated observation geometry consistent.

**Architecture:** Change only the Isaac `DynamicSphere` radius and the configured 2D tennis-ball box dimensions. Preserve object centers, mass, color, fixed-grid control, YOLO class labels, IK, gripper, and attachment parameters.

**Tech Stack:** Isaac Sim Python, ROS 2 YAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-experiment3-circular-sorting-design.md`

## Global Constraints

- Modify only `实验三`.
- Use physical radius `0.020 m`.
- Use deterministic observation boxes `40 x 40` pixels.
- Keep every box center unchanged.
- Do not modify mass, color, coordinates, IK, gripper, or attachment settings.
- Do not run Git commands.

---

### Task 1: Reduce tennis-ball geometry

**Files:**
- Modify: `simulation/isaac/experiment3_scene.py`
- Modify: `config/experiment3_sorting.yaml`
- Modify: `tests/test_experiment3_config.py`

**Interfaces:**
- Consumes: the existing `tennis_ball` object definitions and their fixed centers.
- Produces: 20 mm Isaac spheres and 40 by 40 pixel observation boxes at unchanged centers.

- [ ] Add a test requiring `radius=0.020` and 40 by 40 tennis-ball boxes.
- [ ] Run the focused test and observe failure against the current 23 mm/46 pixel values.
- [ ] Change the scene radius and all tennis-ball observation dimensions.
- [ ] Run Python syntax checks and the complete lightweight test suite.

## Success Criteria

- Every tennis-ball object has physical radius `0.020 m`.
- Every configured tennis-ball bbox has width and height `40`.
- Tennis-ball centers and all motion parameters remain unchanged.
- All lightweight tests pass.
