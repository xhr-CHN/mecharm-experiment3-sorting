"""Isaac Sim fixtures for the circular tennis-ball and pencil sorter."""

from __future__ import annotations

from typing import Mapping


def build_experiment3_fixtures(world, stage, config: Mapping) -> None:
    """Create a centered table, eight-position ring, and an overhead camera."""
    import numpy as np
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.api.objects import DynamicSphere, FixedCuboid

    root = "/World/experiment3"
    center = tuple(float(value) for value in config["ring_center"])
    pencil_material = UsdShade.Material.Define(
        stage, f"{root}/materials/pencil_grip_material"
    )
    physics_material = UsdPhysics.MaterialAPI.Apply(pencil_material.GetPrim())
    physics_material.CreateStaticFrictionAttr().Set(1.0)
    physics_material.CreateDynamicFrictionAttr().Set(1.0)
    physics_material.CreateRestitutionAttr().Set(0.0)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(pencil_material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set("max")
    for jaw_path in (
        "/World/mecharm_270_pi/gripper_left1",
        "/World/mecharm_270_pi/gripper_right1",
    ):
        jaw = stage.GetPrimAtPath(jaw_path)
        if jaw and jaw.IsValid():
            UsdShade.MaterialBindingAPI(jaw).Bind(pencil_material)
    world.scene.add(
        FixedCuboid(
            prim_path=f"{root}/work_table",
            name="experiment3_work_table",
            position=np.array([center[0], center[1], -0.025]),
            scale=np.array([0.65, 0.65, 0.05]),
            color=np.array([0.42, 0.28, 0.16]),
        )
    )

    for grid_id, point in config["grid_centers"].items():
        x, y, _ = (float(value) for value in point)
        world.scene.add(
            FixedCuboid(
                prim_path=f"{root}/grid_markers/{grid_id}",
                name=f"grid_marker_{grid_id}",
                position=np.array([x, y, 0.003]),
                scale=np.array([0.075, 0.075, 0.006]),
                color=np.array([0.12, 0.12, 0.12]),
            )
        )

    for item in config["objects"]:
        object_id = str(item["object_id"])
        class_id = str(item["class_id"])
        x, y, z = (float(value) for value in item["center"])
        prim_path = f"{root}/objects/{object_id}"
        if class_id == "tennis_ball":
            world.scene.add(
                DynamicSphere(
                    prim_path=prim_path,
                    name=object_id,
                    position=np.array([x, y, z]),
                    radius=0.020,
                    color=np.array([0.68, 0.90, 0.08]),
                    mass=0.035,
                )
            )
        elif class_id == "pencil":
            pencil = UsdGeom.Xform.Define(stage, prim_path)
            UsdGeom.XformCommonAPI(pencil).SetTranslate((x, y, z))
            UsdPhysics.RigidBodyAPI.Apply(pencil.GetPrim())
            UsdPhysics.MassAPI.Apply(pencil.GetPrim()).CreateMassAttr().Set(0.018)

            shaft = UsdGeom.Cylinder.Define(stage, f'{prim_path}/shaft')
            shaft.CreateAxisAttr().Set(UsdGeom.Tokens.x)
            shaft.CreateHeightAttr().Set(0.090)
            shaft.CreateRadiusAttr().Set(0.0075)
            shaft.CreateDisplayColorAttr().Set([Gf.Vec3f(0.08, 0.55, 0.18)])
            UsdPhysics.CollisionAPI.Apply(shaft.GetPrim())
            UsdShade.MaterialBindingAPI(shaft.GetPrim()).Bind(pencil_material)

            wood_tip = UsdGeom.Cone.Define(stage, f'{prim_path}/wood_tip')
            wood_tip.CreateAxisAttr().Set(UsdGeom.Tokens.x)
            wood_tip.CreateHeightAttr().Set(0.0225)
            wood_tip.CreateRadiusAttr().Set(0.0075)
            wood_tip.CreateDisplayColorAttr().Set([Gf.Vec3f(0.76, 0.56, 0.30)])
            UsdGeom.XformCommonAPI(wood_tip).SetTranslate((0.05625, 0.0, 0.0))
            UsdPhysics.CollisionAPI.Apply(wood_tip.GetPrim())

            graphite_tip = UsdGeom.Cone.Define(stage, f'{prim_path}/graphite_tip')
            graphite_tip.CreateAxisAttr().Set(UsdGeom.Tokens.x)
            graphite_tip.CreateHeightAttr().Set(0.0075)
            graphite_tip.CreateRadiusAttr().Set(0.0028125)
            graphite_tip.CreateDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.05, 0.05)])
            # Overlap the wood cone by 1 mm so the graphite tip renders and
            # collides as one continuous sharpened section without a gap.
            UsdGeom.XformCommonAPI(graphite_tip).SetTranslate((0.0638, 0.0, 0.0))
            UsdPhysics.CollisionAPI.Apply(graphite_tip.GetPrim())
        else:
            raise ValueError(f"unsupported experiment3 class: {class_id}")

    bin_outer_size = 0.13
    bin_wall_thickness = 0.008
    bin_wall_height = 0.045
    bin_bottom_thickness = 0.012
    for bin_id, point in config["class_bins"].items():
        x, y, _ = (float(value) for value in point)
        bin_root = f"{root}/bins/{bin_id}"
        color = np.array(
            [0.15, 0.35, 0.85] if bin_id == "tennis_ball" else [0.85, 0.25, 0.10]
        )
        parts = (
            ("bottom", (x, y, bin_bottom_thickness / 2.0),
             (bin_outer_size, bin_outer_size, bin_bottom_thickness)),
            ("wall_pos_x", (x + (bin_outer_size - bin_wall_thickness) / 2.0, y,
                            bin_bottom_thickness + bin_wall_height / 2.0),
             (bin_wall_thickness, bin_outer_size, bin_wall_height)),
            ("wall_neg_x", (x - (bin_outer_size - bin_wall_thickness) / 2.0, y,
                            bin_bottom_thickness + bin_wall_height / 2.0),
             (bin_wall_thickness, bin_outer_size, bin_wall_height)),
            ("wall_pos_y", (x, y + (bin_outer_size - bin_wall_thickness) / 2.0,
                            bin_bottom_thickness + bin_wall_height / 2.0),
             (bin_outer_size - 2.0 * bin_wall_thickness, bin_wall_thickness, bin_wall_height)),
            ("wall_neg_y", (x, y - (bin_outer_size - bin_wall_thickness) / 2.0,
                            bin_bottom_thickness + bin_wall_height / 2.0),
             (bin_outer_size - 2.0 * bin_wall_thickness, bin_wall_thickness, bin_wall_height)),
        )
        for part_name, position, scale in parts:
            world.scene.add(
                FixedCuboid(
                    prim_path=f"{bin_root}/{part_name}",
                    name=f"bin_{bin_id}_{part_name}",
                    position=np.array(position),
                    scale=np.array(scale),
                    color=color,
                )
            )

    camera_path = f"{root}/top_camera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    # 18 mm keeps the full 0.22 m ring inside the 848x480 vertical field of view.
    camera.CreateFocalLengthAttr().Set(18.0)
    camera.CreateHorizontalApertureAttr().Set(20.955)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 5.0))
    # Camera sweep 20260909-233453 selected a 10-degree north tilt (5/5 at 6/6).
    camera_position = Gf.Vec3d(center[0], center[1] + 0.1939596787793115, 1.10)
    camera_to_world = Gf.Matrix4d().SetLookAt(
        camera_position,
        Gf.Vec3d(center[0], center[1], 0.0),
        Gf.Vec3d(0.0, 1.0, 0.0),
    ).GetInverse()
    UsdGeom.Xformable(camera).AddTransformOp().Set(camera_to_world)
    print(f"Created experiment3 centered ring, open bins, and camera: {camera_path}", flush=True)
