"""Start the Isaac Sim adapter and ROS 2 task/recorder nodes."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_linux_root = "/mnt/e/机器人集成小组项目/实验二"
    default_windows_root = r"E:\机器人集成小组项目\实验二"
    config = LaunchConfiguration("config")
    result_root = LaunchConfiguration("result_root")
    start_isaac = LaunchConfiguration("start_isaac")
    windows_root = LaunchConfiguration("windows_root")

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
        ],
        output="screen",
        condition=IfCondition(start_isaac),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=str(Path(default_linux_root) / "config/simulation.yaml")),
            DeclareLaunchArgument("result_root", default_value=str(Path(default_linux_root) / "results/simulation")),
            DeclareLaunchArgument("windows_root", default_value=default_windows_root),
            DeclareLaunchArgument("start_isaac", default_value="true"),
            isaac_process,
            Node(
                package="mecharm_pick_place",
                executable="pick_place_task_node",
                name="mecharm_pick_place",
                parameters=[config],
                output="screen",
            ),
            Node(
                package="mecharm_pick_place",
                executable="pick_place_recorder_node",
                name="pick_place_recorder_node",
                parameters=[{"result_root": result_root, "config_path": config}],
                output="screen",
            ),
        ]
    )
