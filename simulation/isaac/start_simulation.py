"""Build and run the mechArm 270 Pi fixed-point pick-and-place scene."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--build-scene-only", action="store_true")
    parser.add_argument("--rebuild-scene", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--gripper-sweep", action="store_true")
    parser.add_argument("--experiment3", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
PROJECT_ROOT = Path(ARGS.project_root).resolve()
SCENE_PATH = PROJECT_ROOT / "simulation/scenes/mecharm_pick_place.usd"

GRIPPER_SWEEP_JOINTS = (
    "gripper_controller",
    "gripper_base_to_gripper_left2",
    "gripper_left3_to_gripper_left1",
    "gripper_base_to_gripper_right3",
    "gripper_base_to_gripper_right2",
    "gripper_right3_to_gripper_right1",
)


def run_gripper_sweep(world, robot, headless: bool) -> None:
    """Sweep the master gripper joint and report both four-bar sides.

    Uses the already-created World/articulation from main(), so it cannot
    conflict with a second simulation context.  The arm is held at its reset
    pose; only gripper_controller is commanded.
    """
    import numpy as np
    from isaacsim.core.utils.types import ArticulationAction

    arm_names = (
        "joint1_to_base",
        "joint2_to_joint1",
        "joint3_to_joint2",
        "joint4_to_joint3",
        "joint5_to_joint4",
        "joint6_to_joint5",
    )
    arm_indices = np.asarray(
        [robot.get_dof_index(name) for name in arm_names], dtype=np.int32
    )
    monitor_indices = {
        name: robot.get_dof_index(name) for name in GRIPPER_SWEEP_JOINTS
    }
    if np.any(arm_indices < 0) or any(
        index is None or index < 0 for index in monitor_indices.values()
    ):
        raise RuntimeError("gripper sweep could not resolve every monitored DOF")

    controller = robot.get_articulation_controller()
    kps, kds = controller.get_gains()
    kps[arm_indices] = 1000.0
    kds[arm_indices] = 100.0
    controller.set_gains(kps=kps, kds=kds)

    all_positions = robot.get_joint_positions()
    if all_positions is None:
        raise RuntimeError("gripper sweep could not read initial joint positions")
    arm_start = np.asarray(
        [float(all_positions[index]) for index in arm_indices], dtype=float
    )
    master_index = int(monitor_indices["gripper_controller"])
    master_start = float(all_positions[master_index])
    gripper_drive_joints = (
        "gripper_controller",
        "gripper_base_to_gripper_left2",
        "gripper_base_to_gripper_right3",
        "gripper_base_to_gripper_right2",
    )
    gripper_indices = np.asarray(
        [monitor_indices[name] for name in gripper_drive_joints],
        dtype=np.int32,
    )

    world.play()
    sweep_hz = 120.0
    segments = (
        ("OPEN", 0.15, 5.0),
        ("CLOSE", -0.75, 7.0),
        ("OPEN", 0.15, 5.0),
    )
    print("Gripper sweep starting", flush=True)
    print(
        "step  master(deg) L2(deg) R2(deg) R3(deg) L_outer(deg) R_outer(deg)",
        flush=True,
    )
    global_step = 0
    for label, target, duration_s in segments:
        start = master_start if label == "OPEN" and global_step == 0 else None
        if start is None:
            start = float(
                all_positions[master_index]
                if global_step == 0
                else robot.get_joint_positions()[master_index]
            )
        duration_steps = int(duration_s * sweep_hz)
        for step_index in range(duration_steps):
            ratio = step_index / duration_steps
            smooth = ratio * ratio * (3.0 - 2.0 * ratio)
            master = start + smooth * (target - start)
            gripper_targets = np.asarray(
                (master, master, -master, -master), dtype=float
            )
            command = np.concatenate((arm_start, gripper_targets))
            indices = np.concatenate((arm_indices, gripper_indices))
            controller.apply_action(
                ArticulationAction(
                    joint_positions=command,
                    joint_indices=indices,
                )
            )
            world.step(render=not headless)
            global_step += 1
            if global_step % 30 == 0:
                measured = robot.get_joint_positions()
                if measured is not None:
                    values = {
                        name: float(measured[index])
                        for name, index in monitor_indices.items()
                    }
                    print(
                        f"{global_step:5d} "
                        f"{np.degrees(values['gripper_controller']):9.2f} "
                        f"{np.degrees(values['gripper_base_to_gripper_left2']):8.2f} "
                        f"{np.degrees(values['gripper_base_to_gripper_right2']):8.2f} "
                        f"{np.degrees(values['gripper_base_to_gripper_right3']):8.2f} "
                        f"{np.degrees(values['gripper_left3_to_gripper_left1']):12.2f} "
                        f"{np.degrees(values['gripper_right3_to_gripper_right1']):12.2f}",
                        flush=True,
                    )
        print(f"Gripper sweep segment finished: {label}", flush=True)
    final = robot.get_joint_positions()
    if final is not None:
        values = {
            name: float(final[index]) for name, index in monitor_indices.items()
        }
        mirror = np.degrees(values["gripper_left3_to_gripper_left1"]) + np.degrees(
            values["gripper_right3_to_gripper_right1"]
        )
        print(
            f"Gripper sweep final |L_outer + R_outer| = {abs(mirror):.2f} deg "
            f"({'SYMMETRIC' if abs(mirror) <= 8.0 else 'ASYMMETRIC'})",
            flush=True,
        )
    print("Gripper sweep completed", flush=True)

# Python 3.11 does not use PATH for dependent DLL lookup on Windows. Keep the
# Isaac ROS 2 runtime directory registered for the lifetime of the process.
ISAAC_SIM_ROOT = Path(os.environ.get("ISAAC_SIM_ROOT", r"E:\AIRobotic\isaac-sim"))
ROS2_RUNTIME = ISAAC_SIM_ROOT / "exts/isaacsim.ros2.bridge/humble/lib"
os.environ.setdefault("ROS_DISTRO", "humble")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
runtime_text = str(ROS2_RUNTIME)
if runtime_text.lower() not in os.environ.get("PATH", "").lower():
    os.environ["PATH"] = runtime_text + os.pathsep + os.environ.get("PATH", "")
ROS2_DLL_HANDLE = os.add_dll_directory(
    runtime_text
)
TRANSPORT = os.environ.get("MECHARM_TRANSPORT", "tcp").strip().lower()
print(
    "Isaac transport environment: "
    f"transport={TRANSPORT}, "
    f"domain={os.environ.get('ROS_DOMAIN_ID', '<unset>')}, "
    f"rmw={os.environ.get('RMW_IMPLEMENTATION', '<unset>')}, "
    f"discovery={os.environ.get('ROS_DISCOVERY_SERVER', '<unset>')}",
    flush=True,
)

simulation_app = SimulationApp({"headless": ARGS.headless})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import create_new_stage, open_stage, save_stage  # noqa: E402


URDF_PATH = (
    PROJECT_ROOT
    / "simulation/urdf/mycobot_description/urdf/mecharm_270_pi/mecharm_270_pi_adaptive_gripper.urdf"
)
ROBOT_PRIM_PATH = "/World/mecharm_270_pi"
ARTICULATION_ROOT_PATH = f"{ROBOT_PRIM_PATH}/root_joint"


def build_scene() -> None:
    """Import the official URDF and create the repeatable experiment fixtures."""
    import numpy as np
    import omni.kit.commands
    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.core.utils.prims import create_prim, move_prim

    enable_extension("isaacsim.asset.importer.urdf")
    simulation_app.update()
    create_new_stage()
    create_prim("/World", "Xform")

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("failed to create the Isaac Sim URDF import configuration")
    import_config.merge_fixed_joints = False
    import_config.fix_base = True
    import_config.make_default_prim = False
    import_config.create_physics_scene = True
    # Do not import URDF mimic tags: PhysX 5.1 silently ignores additional
    # mimic followers of the same reference joint, so the right gripper side
    # receives no command.  Every inner gripper link is instead driven by its
    # own explicit angular drive, and the TCP bridge mirrors gripper_controller
    # onto the followers in software.
    import_config.parse_mimic = False
    self_collision_mode = os.environ.get(
        "MECHARM_SELF_COLLISION", "selective"
    ).strip().lower()
    if self_collision_mode not in ("off", "selective", "all"):
        raise RuntimeError(
            "MECHARM_SELF_COLLISION must be off, selective, or all"
        )
    import_config.set_self_collision(self_collision_mode != "off")
    import_config.set_default_drive_strength(1000)
    import_config.set_default_position_drive_damping(30000)
    import_config.set_collision_from_visuals(True)
    # The 5.1 URDF importer cannot parse a source path containing Chinese
    # characters. Stage the same checked-in assets under a temporary ASCII path.
    source_description = URDF_PATH.parents[2]
    with tempfile.TemporaryDirectory(prefix="mecharm_exp3_") as temp_root:
        staged_description = Path(temp_root) / "mycobot_description"
        shutil.copytree(source_description, staged_description)
        staged_urdf = staged_description / URDF_PATH.relative_to(source_description)
        previous_cwd = Path.cwd()
        try:
            os.chdir(temp_root)
            status, imported_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=str(staged_urdf),
                import_config=import_config,
            )
        finally:
            os.chdir(previous_cwd)
    if not status or not imported_path:
        raise RuntimeError(f"failed to import mechArm URDF: {URDF_PATH}")
    if imported_path != ROBOT_PRIM_PATH:
        move_prim(imported_path, ROBOT_PRIM_PATH)

    # Isaac Sim 5.1 may clear limits from imported mimic targets even when the
    # URDF contains finite limits. PhysX refuses to activate mimic joints
    # without finite angular limits, so restore them before the first reset.
    stage = omni.usd.get_context().get_stage()

    if self_collision_mode == "selective":
        arm_links = ("base", "link1", "link2", "link3", "link4", "link5", "link6")
        gripper_links = (
            "gripper_base",
            "gripper_left1",
            "gripper_left2",
            "gripper_left3",
            "gripper_right1",
            "gripper_right2",
            "gripper_right3",
        )
        filtered_pairs = {
            ("base", "link1"),
            ("link1", "link2"),
            ("link2", "link3"),
            ("link3", "link4"),
            ("link4", "link5"),
            ("link5", "link6"),
            ("link6", "gripper_base"),
        }
        filtered_pairs.update(
            (arm_link, gripper_link)
            for arm_link in arm_links
            for gripper_link in gripper_links
        )
        filtered_pairs.update(
            (first, second)
            for index, first in enumerate(gripper_links)
            for second in gripper_links[index + 1 :]
            if {first, second} != {"gripper_left1", "gripper_right1"}
        )
        for first, second in sorted(filtered_pairs):
            first_path = f"{ROBOT_PRIM_PATH}/{first}"
            second_path = f"{ROBOT_PRIM_PATH}/{second}"
            first_prim = stage.GetPrimAtPath(first_path)
            second_prim = stage.GetPrimAtPath(second_path)
            if not first_prim.IsValid() or not second_prim.IsValid():
                raise RuntimeError(
                    f"cannot configure selective self-collision: {first_path}, {second_path}"
                )
            filtering = UsdPhysics.FilteredPairsAPI.Apply(first_prim)
            filtering.CreateFilteredPairsRel().AddTarget(Sdf.Path(second_path))
        print(
            "Configured selective robot self-collision; opposing final jaws remain active",
            flush=True,
        )
    driven_inner_limits = {
        "gripper_base_to_gripper_left2": (-42.972, 8.594),
        "gripper_base_to_gripper_right3": (-8.594, 42.972),
        "gripper_base_to_gripper_right2": (-8.594, 42.972),
    }
    for joint_name, (lower, upper) in driven_inner_limits.items():
        joint_path = f"{ROBOT_PRIM_PATH}/joints/{joint_name}"
        joint = UsdPhysics.RevoluteJoint.Get(stage, joint_path)
        if not joint:
            raise RuntimeError(f"missing adaptive-gripper inner joint: {joint_path}")
        joint.GetLowerLimitAttr().Set(lower)
        joint.GetUpperLimitAttr().Set(upper)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.GetStiffnessAttr().Set(150.0)
        drive.GetDampingAttr().Set(60.0)
        drive.GetMaxForceAttr().Set(20.0)
    print("Configured direct-driven adaptive-gripper inner joints", flush=True)

    # The URDF importer gives every revolute joint a very strong default drive.
    # That is useful for the arm, but it makes the closed gripper loop hunt
    # around its pin anchors.  Use a softer, critically damped drive only for
    # the commanded gripper joints; the outer jaw joints remain passive.
    controller_path = f"{ROBOT_PRIM_PATH}/joints/gripper_controller"
    controller = UsdPhysics.RevoluteJoint.Get(stage, controller_path)
    if not controller:
        raise RuntimeError(f"missing adaptive-gripper controller joint: {controller_path}")
    controller_drive = UsdPhysics.DriveAPI.Apply(controller.GetPrim(), "angular")
    controller_drive.GetStiffnessAttr().Set(150.0)
    controller_drive.GetDampingAttr().Set(60.0)
    controller_drive.GetMaxForceAttr().Set(20.0)
    print("Tuned adaptive-gripper drive damping", flush=True)

    # The official URDF is a tree and therefore omits the two distal pin
    # joints that close the adaptive gripper's four-bar loops.  Make the outer
    # jaw joints passive and add the missing PhysX loop constraints explicitly.
    for passive_name in (
        "gripper_left3_to_gripper_left1",
        "gripper_right3_to_gripper_right1",
    ):
        passive_path = f"{ROBOT_PRIM_PATH}/joints/{passive_name}"
        passive_joint = UsdPhysics.RevoluteJoint.Get(stage, passive_path)
        if not passive_joint:
            raise RuntimeError(f"missing passive adaptive-gripper joint: {passive_path}")
        passive_prim = passive_joint.GetPrim()
        passive_prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
        # The URDF importer may leave the drive as plain physics attributes
        # (not an applied DriveAPI), so RemoveAPI alone silently keeps them.
        # Delete every remaining drive attribute; otherwise the outer jaw is
        # both pulled by the four-bar loop and spring-driven back to 0, which
        # makes that side feel loose and unable to exert force.
        for attribute in list(passive_prim.GetAttributes()):
            if attribute.GetName().startswith("drive:"):
                passive_prim.RemoveProperty(attribute.GetName())
        remaining_drives = [
            attribute.GetName()
            for attribute in passive_prim.GetAttributes()
            if attribute.GetName().startswith("drive:")
        ]
        if remaining_drives:
            raise RuntimeError(
                f"passive adaptive-gripper joint still has a drive: {passive_path}: "
                f"{remaining_drives}"
            )

    print("Converted outer gripper joints to passive revolute joints", flush=True)

    loop_joints = (
        (
            "gripper_left_loop_joint",
            f"{ROBOT_PRIM_PATH}/gripper_left2",
            f"{ROBOT_PRIM_PATH}/gripper_left1",
            (-0.027963, 0.015311, 0.0),
            (0.006037, 0.021311, 0.0),
        ),
        (
            "gripper_right_loop_joint",
            f"{ROBOT_PRIM_PATH}/gripper_right2",
            f"{ROBOT_PRIM_PATH}/gripper_right1",
            (0.027963, 0.015311, 0.0),
            (-0.006037, 0.021311, 0.0),
        ),
    )
    for joint_name, body0, body1, local0, local1 in loop_joints:
        joint_path = f"{ROBOT_PRIM_PATH}/joints/{joint_name}"
        loop_joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        loop_joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
        loop_joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        loop_joint.CreateAxisAttr().Set(UsdPhysics.Tokens.z)
        loop_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local0))
        loop_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local1))
        loop_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        loop_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        loop_joint.CreateCollisionEnabledAttr().Set(False)
        loop_joint.CreateExcludeFromArticulationAttr().Set(True)
        if len(loop_joint.GetBody0Rel().GetTargets()) != 1 or len(loop_joint.GetBody1Rel().GetTargets()) != 1:
            raise RuntimeError(f"failed to create adaptive-gripper loop joint: {joint_path}")

    articulation_api = PhysxSchema.PhysxArticulationAPI.Get(
        stage, ARTICULATION_ROOT_PATH
    )
    if not articulation_api:
        articulation_api = PhysxSchema.PhysxArticulationAPI.Get(
            stage, ROBOT_PRIM_PATH
        )
    if not articulation_api:
        raise RuntimeError("missing PhysX articulation API for mechArm")
    articulation_api.CreateSolverPositionIterationCountAttr().Set(64)
    articulation_api.CreateSolverVelocityIterationCountAttr().Set(64)
    print(
        "Created left/right adaptive-gripper loop pin constraints outside articulation",
        flush=True,
    )

    # The official adaptive-gripper DAE files contain the linkage plates and
    # pivot holes, but no separate bolt/pin meshes.  PhysX revolute joints are
    # the actual constraints; these cylinders make the four exposed pivots
    # visually complete without adding collision geometry that could jam them.
    exposed_pins = (
        (
            "left_base_pin",
            f"{ROBOT_PRIM_PATH}/gripper_base",
            (-0.005, 0.027, -0.012),
            "gripper_base_to_gripper_left2",
        ),
        (
            "left_tip_pin",
            f"{ROBOT_PRIM_PATH}/gripper_left1",
            (0.006037, 0.021311, -0.012),
            "gripper_left_loop_joint",
        ),
        (
            "right_base_pin",
            f"{ROBOT_PRIM_PATH}/gripper_base",
            (0.005, 0.027, -0.012),
            "gripper_base_to_gripper_right2",
        ),
        (
            "right_tip_pin",
            f"{ROBOT_PRIM_PATH}/gripper_right1",
            (-0.006037, 0.021311, -0.012),
            "gripper_right_loop_joint",
        ),
    )
    for pin_name, parent_path, local_position, joint_name in exposed_pins:
        joint_path = f"{ROBOT_PRIM_PATH}/joints/{joint_name}"
        joint = UsdPhysics.RevoluteJoint.Get(stage, joint_path)
        if not joint:
            raise RuntimeError(f"missing adaptive-gripper pivot joint: {joint_path}")
        if len(joint.GetBody0Rel().GetTargets()) != 1 or len(joint.GetBody1Rel().GetTargets()) != 1:
            raise RuntimeError(f"incomplete rigid-body connection at pivot: {joint_path}")
        parent = stage.GetPrimAtPath(parent_path)
        if not parent or not parent.IsValid():
            raise RuntimeError(f"missing adaptive-gripper pin parent: {parent_path}")
        pin_path = f"{parent_path}/physics_pins/{pin_name}"
        pin = UsdGeom.Cylinder.Define(stage, pin_path)
        pin.CreateAxisAttr().Set(UsdGeom.Tokens.z)
        pin.CreateRadiusAttr().Set(0.0032)
        pin.CreateHeightAttr().Set(0.026)
        pin.CreateDisplayColorAttr().Set([(0.32, 0.34, 0.38)])
        UsdGeom.XformCommonAPI(pin).SetTranslate(local_position)

    world = World(stage_units_in_meters=1.0)
    if ARGS.experiment3:
        import yaml

        config_path = PROJECT_ROOT / "config/experiment3_sorting.yaml"
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)["sorting_task"]["ros__parameters"]
        from experiment3_scene import build_experiment3_fixtures

        build_experiment3_fixtures(world, stage, config)
    else:
        world.scene.add(
            FixedCuboid(
                prim_path="/World/work_table",
                name="work_table",
                position=np.array([0.14, 0.0, -0.025]),
                scale=np.array([0.60, 0.50, 0.05]),
                color=np.array([0.55, 0.40, 0.25]),
            )
        )
        world.scene.add(
            DynamicCuboid(
                prim_path="/World/target_object",
                name="target_object",
                position=np.array([0.18, 0.08, 0.025]),
                scale=np.array([0.035, 0.035, 0.05]),
                color=np.array([0.85, 0.15, 0.10]),
                mass=0.03,
            )
        )
    world.reset()
    SCENE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not save_stage(str(SCENE_PATH)):
        raise RuntimeError(f"failed to save Isaac scene: {SCENE_PATH}")
    print(f"Saved mechArm experiment scene: {SCENE_PATH}", flush=True)


def main() -> None:
    print(
        "Isaac scene request: "
        f"rebuild={ARGS.rebuild_scene}, build_only={ARGS.build_scene_only}, "
        f"scene={SCENE_PATH}",
        flush=True,
    )
    if TRANSPORT == "ros2":
        enable_extension("isaacsim.ros2.bridge")
        for _ in range(30):
            simulation_app.update()
    elif TRANSPORT != "tcp":
        raise RuntimeError(f"unsupported MECHARM_TRANSPORT: {TRANSPORT}")

    if ARGS.rebuild_scene or not SCENE_PATH.is_file():
        build_scene()
    if ARGS.build_scene_only:
        return

    if not open_stage(str(SCENE_PATH)):
        raise RuntimeError(f"failed to open Isaac scene: {SCENE_PATH}")
    for _ in range(10):
        simulation_app.update()

    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from grasp_monitor import GraspMonitor

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
    robot = SingleArticulation(prim_path=ARTICULATION_ROOT_PATH, name="mecharm_270_pi")
    world.scene.add(robot)
    world.reset()
    robot.initialize()
    camera_stream = None
    if ARGS.experiment3:
        from experiment3_camera_stream import IsaacCameraTcpStream

        camera_stream = IsaacCameraTcpStream(
            "/World/experiment3/top_camera",
            width=848,
            height=480,
            bind_host=os.environ.get("MECHARM_CAMERA_BIND", "0.0.0.0"),
            port=int(os.environ.get("MECHARM_CAMERA_PORT", "8766")),
            publish_rate_hz=10.0,
        )
    if ARGS.gripper_sweep:
        run_gripper_sweep(world, robot, headless=ARGS.headless)
        return
    tcp_bridge = None
    if TRANSPORT == "tcp":
        from tcp_joint_bridge import IsaacTcpJointBridge

        tcp_bridge = IsaacTcpJointBridge(
            robot,
            bind_host=os.environ.get("MECHARM_TCP_BIND", "0.0.0.0"),
            port=int(os.environ.get("MECHARM_TCP_PORT", "8765")),
        )
    else:
        from ros2_graph import create_ros2_graph

        create_ros2_graph(ARTICULATION_ROOT_PATH)
        for _ in range(10):
            simulation_app.update()
    monitor_kwargs = {}
    if ARGS.experiment3:
        monitor_kwargs["object_root_path"] = "/World/experiment3/objects"
        monitor_kwargs["joint_path"] = "/World/experiment3/grasp_fixed_joint"
    monitor = GraspMonitor(omni.usd.get_context().get_stage(), robot, **monitor_kwargs)
    max_steps = 30 if ARGS.smoke_test else None
    print(
        "Starting Isaac control loop: "
        f"transport={TRANSPORT}, "
        f"headless={ARGS.headless}, smoke_test={ARGS.smoke_test}, max_steps={max_steps}"
    )
    world.play()
    steps = 0
    try:
        while not simulation_app.is_exiting():
            if tcp_bridge is not None:
                tcp_bridge.apply_latest_command()
            world.step(render=ARGS.experiment3 or not ARGS.headless)
            if camera_stream is not None:
                camera_stream.update()
            if tcp_bridge is not None:
                tcp_bridge.publish_state()
            else:
                from ros2_graph import tick_ros2_graph

                tick_ros2_graph()
            monitor.update()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                print(f"Isaac control smoke test completed: {steps} simulation steps")
                break
    finally:
        if tcp_bridge is not None:
            tcp_bridge.close()
        if camera_stream is not None:
            camera_stream.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        print("Isaac experiment failed with an exception:", flush=True)
        traceback.print_exc()
        simulation_app.close()
        os._exit(1)
    else:
        simulation_app.close()
