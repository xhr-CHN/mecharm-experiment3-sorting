from glob import glob
from setuptools import find_packages, setup


package_name = "mecharm_pick_place"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="xhr",
    maintainer_email="xhr@example.com",
    description="Fixed-point pick-and-place task for mechArm 270 Pi.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "pick_place_task_node = mecharm_pick_place.task_node:main",
            "pick_place_recorder_node = mecharm_pick_place.recorder_node:main",
            "isaac_trajectory_controller = mecharm_pick_place.trajectory_controller_node:main",
            "isaac_tcp_bridge = mecharm_pick_place.tcp_bridge_node:main",
            "direct_motion_probe = mecharm_pick_place.direct_motion_probe:main",
            "direct_pick_place_demo = mecharm_pick_place.direct_pick_place_demo:main",
            "cartesian_direct_pick_place = mecharm_pick_place.cartesian_direct_pick_place:main",
            "sim_object_detector = mecharm_pick_place.sim_object_detector:main",
            "sim_camera_node = mecharm_pick_place.sim_camera_node:main",
            "sorting_task_node = mecharm_pick_place.sorting_task_node:main",
            "virtual_camera_node = mecharm_pick_place.virtual_camera_node:main",
            "yolo_classifier_node = mecharm_pick_place.yolo_classifier_node:main",
        ],
    },
)
