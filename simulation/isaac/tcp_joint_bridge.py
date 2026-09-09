"""TCP transport between native Windows Isaac Sim and containerized ROS 2."""

from __future__ import annotations

import json
import math
import socket
import threading
import time

import numpy as np
from isaacsim.core.utils.types import ArticulationAction


JOINT_NAMES = (
    "joint1_to_base",
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "gripper_controller",
    # The articulation tree exposes the mimic followers and the passive outer
    # jaws as DOFs as well. Publishing them lets a manual open/close test
    # verify both four-bar sides from ROS instead of creating a second World
    # inside Isaac (which conflicts with the running control loop and crashes
    # the app).
    "gripper_base_to_gripper_left2",
    "gripper_left3_to_gripper_left1",
    "gripper_base_to_gripper_right3",
    "gripper_base_to_gripper_right2",
    "gripper_right3_to_gripper_right1",
)


class IsaacTcpJointBridge:
    def __init__(self, articulation, bind_host="0.0.0.0", port=8765):
        self._articulation = articulation
        self._controller = articulation.get_articulation_controller()
        self._indices = np.asarray(
            [articulation.get_dof_index(name) for name in JOINT_NAMES], dtype=np.int32
        )
        if np.any(self._indices < 0):
            raise RuntimeError("TCP bridge could not resolve every controlled joint")
        self._latest_command = None
        self._state_debug_logged = False
        self._state_client_debug_logged = False
        self._command_lock = threading.Lock()
        self._client = None
        self._client_lock = threading.Lock()
        self._stop = threading.Event()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((bind_host, int(port)))
        self._server.listen(1)
        self._server.settimeout(0.5)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        print(f"Isaac TCP joint bridge listening on {bind_host}:{port}", flush=True)

    @staticmethod
    def _mirrored_gripper_targets(names, positions):
        """Expand a gripper_controller command to all driven inner links.

        PhysX mimic followers of the same reference joint are unreliable in
        this scene (the right side is silently ignored), so the scene drives
        every inner gripper link explicitly.  The ROS side only commands
        gripper_controller; this bridge mirrors it: left2 follows with +1 and
        right2/right3 follow with -1.
        """
        if "gripper_controller" not in names:
            return names, positions
        master = float(positions[names.index("gripper_controller")])
        targets = {
            "gripper_controller": master,
            "gripper_base_to_gripper_left2": master,
            "gripper_base_to_gripper_right3": -master,
            "gripper_base_to_gripper_right2": -master,
        }
        expanded_names = list(names)
        expanded_positions = list(positions)
        for joint_name, target in targets.items():
            if joint_name in expanded_names:
                expanded_positions[expanded_names.index(joint_name)] = target
            else:
                expanded_names.append(joint_name)
                expanded_positions.append(target)
        return tuple(expanded_names), tuple(expanded_positions)

    def _serve(self):
        while not self._stop.is_set():
            try:
                client, address = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.settimeout(0.5)
            with self._client_lock:
                previous = self._client
                self._client = client
            if previous is not None:
                try:
                    previous.close()
                except OSError:
                    pass
            print(f"Isaac TCP joint bridge connected: {address}", flush=True)
            self._receive(client)

    def _receive(self, client):
        buffer = b""
        last_message_time = time.monotonic()
        while not self._stop.is_set():
            try:
                chunk = client.recv(65536)
            except socket.timeout:
                if time.monotonic() - last_message_time > 3.0:
                    break
                continue
            except OSError:
                break
            if not chunk:
                break
            last_message_time = time.monotonic()
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                self._accept_message(line)
        with self._client_lock:
            if self._client is client:
                self._client = None
        try:
            client.close()
        except OSError:
            pass
        print("Isaac TCP joint bridge disconnected", flush=True)

    def _accept_message(self, line):
        try:
            message = json.loads(line.decode("utf-8"))
            if message.get("type") != "command":
                return
            names = tuple(message["names"])
            positions = tuple(float(value) for value in message["positions"])
            if len(names) != len(positions) or len(names) == 0:
                return
            if len(names) != len(set(names)) or not set(names).issubset(JOINT_NAMES):
                return
            if not all(math.isfinite(value) for value in positions):
                return
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return
        with self._command_lock:
            self._latest_command = (names, positions)

    def apply_latest_command(self):
        with self._command_lock:
            command = self._latest_command
            self._latest_command = None
        if command is None:
            return
        names, positions = command
        names, positions = self._mirrored_gripper_targets(names, positions)
        indices = np.asarray(
            [self._articulation.get_dof_index(name) for name in names], dtype=np.int32
        )
        self._controller.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(positions, dtype=float),
                joint_indices=indices,
            )
        )

    def publish_state(self):
        positions = self._articulation.get_joint_positions()
        if positions is None:
            if not self._state_debug_logged:
                print("Isaac TCP state skipped: articulation returned no joint positions", flush=True)
                self._state_debug_logged = True
            return
        message = {
            "type": "state",
            "names": JOINT_NAMES,
            "positions": [float(positions[index]) for index in self._indices],
        }
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        with self._client_lock:
            client = self._client
        if client is None:
            if not self._state_client_debug_logged:
                print("Isaac TCP state skipped: no ROS TCP client", flush=True)
                self._state_client_debug_logged = True
            return
        try:
            client.sendall(payload)
            if not self._state_debug_logged:
                print(f"Isaac TCP state streaming: {len(message['positions'])} joints", flush=True)
                self._state_debug_logged = True
        except OSError:
            with self._client_lock:
                if self._client is client:
                    self._client = None

    def close(self):
        self._stop.set()
        with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            try:
                client.close()
            except OSError:
                pass
        try:
            self._server.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)
