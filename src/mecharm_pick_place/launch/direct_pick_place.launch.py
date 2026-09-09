from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
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
                    executable="direct_pick_place_demo",
                    name="direct_pick_place_demo",
                    output="screen",
                )
            ],
        ),
    ])
