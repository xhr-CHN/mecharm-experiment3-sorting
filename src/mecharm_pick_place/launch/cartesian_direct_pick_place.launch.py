from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    from pathlib import Path

    package_share = Path(get_package_share_directory("mecharm_pick_place"))
    return LaunchDescription([
        DeclareLaunchArgument("project_root", default_value="/workspace/mecharm_exp2"),
        Node(
            package="mecharm_pick_place",
            executable="isaac_tcp_bridge",
            name="isaac_tcp_bridge",
            output="screen",
            parameters=[[project_root, "/config/simulation.yaml"]],
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package="mecharm_pick_place",
                    executable="cartesian_direct_pick_place",
                    name="cartesian_direct_pick_place",
                    output="screen",
                    parameters=[str(package_share / "config/direct_cartesian_pick_place.yaml")],
                )
            ],
        ),
    ])
