"""Programmatic Isaac Sim ROS 2 joint-state bridge."""

import omni.graph.core as og
import usdrt.Sdf


GRAPH_PATH = "/ActionGraph"


def create_ros2_graph(robot_prim_path: str) -> str:
    og.Controller.edit(
        {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("Context.inputs:useDomainIDEnvVar", True),
                ("PublishJointState.inputs:topicName", "/joint_states"),
                ("SubscribeJointState.inputs:topicName", "/mecharm/joint_target"),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(robot_prim_path)]),
                ("ArticulationController.inputs:robotPath", robot_prim_path),
            ],
        },
    )
    for node_name in ("PublishJointState", "SubscribeJointState", "ArticulationController"):
        if og.Controller.node(f"{GRAPH_PATH}/{node_name}") is None:
            raise RuntimeError(f"Isaac ROS 2 graph node was not created: {node_name}")
    return GRAPH_PATH


def tick_ros2_graph() -> None:
    og.Controller.set(
        og.Controller.attribute(f"{GRAPH_PATH}/OnImpulseEvent.state:enableImpulse"),
        True,
    )
    og.Controller.evaluate_sync(og.Controller.graph(GRAPH_PATH))
