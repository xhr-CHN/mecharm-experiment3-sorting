"""Launch the complete local experiment-three sorting graph."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_linux_root = "/workspace/mecharm_exp3"
    default_windows_root = r"E:\机器人集成小组项目\实验三"
    project_root = LaunchConfiguration("project_root")
    windows_root = LaunchConfiguration("windows_root")
    config = LaunchConfiguration("config")
    result_root = LaunchConfiguration("result_root")
    start_isaac = LaunchConfiguration("start_isaac")

    isaac_process = ExecuteProcess(
        cmd=[
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            [windows_root, r"\scripts\start_isaac.ps1"],
            "-ProjectRoot",
            windows_root,
            "-Experiment3",
        ],
        output="screen",
        condition=IfCondition(start_isaac),
    )

    simulation_config = [project_root, "/config/simulation.yaml"]
    return LaunchDescription(
        [
            DeclareLaunchArgument("project_root", default_value=default_linux_root),
            DeclareLaunchArgument("windows_root", default_value=default_windows_root),
            DeclareLaunchArgument(
                "config",
                default_value=str(Path(default_linux_root) / "config/experiment3_sorting.yaml"),
            ),
            DeclareLaunchArgument(
                "result_root",
                default_value=str(Path(default_linux_root) / "results/experiment3"),
            ),
            DeclareLaunchArgument("start_isaac", default_value="true"),
            isaac_process,
            Node(
                package="mecharm_pick_place",
                executable="isaac_tcp_bridge",
                name="isaac_tcp_bridge",
                output="screen",
                parameters=[simulation_config],
            ),
            Node(
                package="mecharm_pick_place",
                executable="isaac_trajectory_controller",
                name="isaac_trajectory_controller",
                output="screen",
                parameters=[simulation_config],
            ),
            Node(
                package="mecharm_pick_place",
                executable="virtual_camera_node",
                name="virtual_camera_node",
                output="screen",
                parameters=[
                    {
                        "host": "127.0.0.1",
                        "port": 8766,
                        "image_topic": "/camera/image_raw",
                    }
                ],
            ),
            Node(
                package="mecharm_pick_place",
                executable="yolo_classifier_node",
                name="yolo_classifier_node",
                output="screen",
                parameters=[
                    {
                        "config_path": config,
                        "request_topic": "/mecharm/vision_request",
                        "detection_topic": "/mecharm/detections",
                        "model_path": [project_root, "/models/pencil_tennis_yolo26n_best.pt"],
                    }
                ],
            ),
            Node(
                package="mecharm_pick_place",
                executable="sorting_task_node",
                name="sorting_task_node",
                output="screen",
                parameters=[
                    {
                        "config_path": config,
                        "auto_start": True,
                        "result_root": result_root,
                        "urdf_path": [
                            project_root,
                            "/simulation/urdf/mycobot_description/urdf/mecharm_270_pi/mecharm_270_pi_adaptive_gripper.urdf",
                        ],
                        "pregrasp_clearance": 0.10,
                        "pick_close_clearance": 0.04,
                        "grasp_yaw_offset_deg": 90.0,
                    }
                ],
            ),
            Node(
                package="mecharm_pick_place",
                executable="pick_place_recorder_node",
                name="pick_place_recorder_node",
                output="screen",
                parameters=[{"result_root": result_root, "config_path": config}],
            ),
        ]
    )
