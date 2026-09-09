# Green Wooden Pencil Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace each thick yellow cuboid pencil with a thin green wooden-pencil model while preserving one-piece physics and the existing sorting behavior.

**Architecture:** Build every pencil as one rigid `UsdGeom.Xform` root at its existing object path. Add a colliding green shaft plus non-colliding wood and graphite cone children, then update the deterministic camera rectangles and detection boxes to match the thinner green silhouette.

**Tech Stack:** Isaac Sim 5.1, USD Python (`UsdGeom`, `UsdPhysics`), ROS 2 YAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-experiment3-circular-sorting-design.md`

## Global Constraints

- Keep each object ID and prim path `/World/experiment3/objects/<object_id>` unchanged.
- Keep pencil mass at `0.018 kg`, world position, class ID, grid ID, and grasp target unchanged.
- Use a 90 mm long, 15 mm diameter green cylinder, a 22.5 mm wood cone, and a 7.5 mm graphite cone; all three sections provide collision geometry.
- Apply collision only to the shaft; both cone children remain visual geometry.
- Do not change mechArm, gripper, IK, dynamics, ring, bin, or tennis-ball settings.
- Do not run Git write or synchronization commands.

---

### Task 1: Replace the pencil primitive and align simulation imagery

**Files:**
- Modify: `实验三/simulation/isaac/experiment3_scene.py`
- Modify: `实验三/config/experiment3_sorting.yaml`
- Modify: `实验三/tests/test_experiment3_config.py`

**Interfaces:**
- Consumes: existing pencil `object_id`, `center`, and class `pencil` from `experiment3_sorting.yaml`.
- Produces: one rigid pencil root with child prims `shaft`, `wood_tip`, and `graphite_tip`; thinner green simulated image rectangles and detection boxes.

- [ ] **Step 1: Add a failing source contract test**

Assert that the scene source defines `UsdPhysics.RigidBodyAPI`, `UsdPhysics.CollisionAPI`, `shaft`, `wood_tip`, `graphite_tip`, dimensions `0.090`, `0.006`, `0.012`, and `0.004`, while the pencil branch no longer creates a `DynamicCuboid`.

- [ ] **Step 2: Run the focused test and verify failure**

Run from `E:\机器人集成小组项目\实验三`:

```powershell
python -m pytest .\tests\test_experiment3_config.py -q
```

Expected: the new pencil model contract fails because the composite USD geometry is absent.

- [ ] **Step 3: Implement the minimal one-rigid-body pencil**

Define a translated `UsdGeom.Xform` at the existing prim path, apply `UsdPhysics.RigidBodyAPI` and a `0.018` kg `UsdPhysics.MassAPI`, define a green `UsdGeom.Cube` shaft with `UsdPhysics.CollisionAPI`, then define wood-colored and graphite-colored `UsdGeom.Cone` children along the local X axis without collision APIs.

- [ ] **Step 4: Align deterministic camera and detection geometry**

Keep each pencil center unchanged, reduce its configured detection height from 22 pixels to 12 pixels, and change the simulated RGB rectangle from yellow to green.

- [ ] **Step 5: Run all lightweight validation**

```powershell
$env:PYTHONPATH='E:\机器人集成小组项目\实验三\src\mecharm_pick_place'
python -m py_compile .\simulation\isaac\experiment3_scene.py
python -m pytest .\tests .\src\mecharm_pick_place\test -q
```

Expected: syntax succeeds and all tests pass. Full Isaac rendering and grasp verification remain user-run acceptance steps.

## Plan Self-Review

- The plan covers visual shape, color, physical unity, mass preservation, collision simplification, and deterministic camera consistency.
- No placeholders or unrelated subsystem changes are included.
- All object paths and controller-facing configuration remain stable.

