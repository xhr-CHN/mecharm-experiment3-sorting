# Safe-Drop Next-Grid Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the experiment-two calibrated speed profile for experiment-three sorting, release objects above the deep bins, and move directly to the next fixed grid.

**Architecture:** Reuse the experiment-two direct-motion timing implementation: fast `28 deg/s` ordinary travel, slow `4 deg/s` grasp/lift/vertical stages, and the existing `0.12 rad/s` gripper. The experiment-three target sequence will perform one initial HOME, lift to `z=0.133 m`, move above the bin, release there, continue without an intermediate HOME, and return HOME once after the final grid.

**Tech Stack:** ROS 2 Python, numerical IK, Isaac Sim TCP bridge, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-experiment3-circular-sorting-design.md`

## Global Constraints

- Preserve experiment-two arm gains, damping, and calibrated motion constants; use the experiment-three-only close-force override of 20.
- Keep fixed grid coordinates and YOLO-only class decisions unchanged.
- Use `carry_clearance=0.108 m`, producing a `z=0.133 m` carry/drop height.
- Do not descend into the bin before releasing.
- Do not return HOME between objects; return HOME once after the final grid.
- Do not modify experiment two or run Git commands.

---

### Task 1: Apply the calibrated motion and safe-drop sequence

**Files:**
- Modify: `src/mecharm_pick_place/mecharm_pick_place/direct_pick_place_demo.py`
- Modify: `src/mecharm_pick_place/mecharm_pick_place/direct_motion_probe.py`
- Modify: `src/mecharm_pick_place/mecharm_pick_place/sorting_task_node.py`
- Modify: `config/experiment3_sorting.yaml`
- Modify: `src/mecharm_pick_place/test/test_direct_pick_place_contract.py`
- Modify: `tests/test_experiment3_launch_contract.py`

- [ ] Copy the validated experiment-two direct-motion timing behavior into the experiment-three package copy.
- [ ] Add `carry_clearance: 0.108` to the sorting configuration.
- [ ] Replace the target sequence with pick pregrasp, slow grasp, slow carry lift, fast bin-above travel, high release, and direct next-target continuation.
- [ ] Remove target-level `PLACE_GRASP` and `RETREAT`, retain the one-time batch HOME, and add one final HOME after the last grid.
- [ ] Update contract tests for `28 deg/s` ordinary travel, `4 deg/s` vertical stages, carry height, and no target-level HOME.
- [ ] Run syntax checks, all unit tests, and static IK checks.

## Success Criteria

- Ordinary travel logs `speed=28.0deg/s` and grasp/lift logs `speed=4.0deg/s`.
- Release occurs at `z=0.133 m` above the bin.
- The next target starts from the released pose without an intermediate HOME.
- All lightweight tests pass.
