"""Expose the native Isaac TCP joint bridge as ordinary ROS 2 topics."""

from __future__ import annotations

import json
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


class IsaacTcpBridgeNode(Node):
    def __init__(self):
        super().__init__("isaac_tcp_bridge")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8765)
        self.declare_parameter("publish_rate_hz", 120.0)
        self._host = str(self.get_parameter("host").value)
        self._port = int(self.get_parameter("port").value)
        self._socket = None
        self._socket_lock = threading.Lock()
        self._latest_state = None
        self._state_sequence = 0
        self._published_sequence = -1
        self._state_debug_logged = False
        self._receive_debug_logged = False
        self._raw_debug_logged = False
        self._parse_debug_logged = False
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._publisher = self.create_publisher(JointState, "/joint_states", state_qos)
        self._subscription = self.create_subscription(
            JointState, "/mecharm/joint_target", self._command_callback, 10
        )
        rate = float(self.get_parameter("publish_rate_hz").value)
        self._timer = self.create_timer(1.0 / rate, self._publish_latest_state)
        self._heartbeat_timer = self.create_timer(1.0, self._send_heartbeat)
        self._thread = threading.Thread(target=self._connection_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"Isaac TCP bridge connecting to {self._host}:{self._port}"
        )

    def _connection_loop(self):
        while not self._stop.is_set():
            try:
                connection = socket.create_connection((self._host, self._port), timeout=2.0)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(0.5)
            except OSError:
                time.sleep(1.0)
                continue
            with self._socket_lock:
                self._socket = connection
            self.get_logger().info("Isaac TCP bridge connected")
            self._receive_states(connection)
            with self._socket_lock:
                if self._socket is connection:
                    self._socket = None
            try:
                connection.close()
            except OSError:
                pass
            if not self._stop.is_set():
                self.get_logger().warning("Isaac TCP bridge disconnected; retrying")

    def _receive_states(self, connection):
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = connection.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            if not self._raw_debug_logged:
                self.get_logger().info(
                    f"received {len(chunk)} raw bytes from Isaac TCP bridge"
                )
                self._raw_debug_logged = True
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                self._accept_state(line)

    def _accept_state(self, line):
        try:
            message = json.loads(line.decode("utf-8"))
            if message.get("type") != "state":
                return
            names = [str(name) for name in message["names"]]
            positions = [float(value) for value in message["positions"]]
            if len(names) != len(positions) or not names:
                return
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            if not self._parse_debug_logged:
                self.get_logger().warning(f"rejected Isaac TCP state payload: {exc}")
                self._parse_debug_logged = True
            return
        with self._state_lock:
            self._latest_state = (names, positions)
            self._state_sequence += 1
        if not self._receive_debug_logged:
            self.get_logger().info(
                f"received Isaac TCP state with {len(names)} joints"
            )
            self._receive_debug_logged = True

    def _publish_latest_state(self):
        with self._state_lock:
            if self._latest_state is None or self._state_sequence == self._published_sequence:
                return
            names, positions = self._latest_state
            sequence = self._state_sequence
        if not self._state_debug_logged:
            self.get_logger().info(
                f"publishing /joint_states with {len(names)} joints"
            )
            self._state_debug_logged = True
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = names
        message.position = positions
        self._publisher.publish(message)
        self._published_sequence = sequence

    def _command_callback(self, message):
        if len(message.name) != len(message.position) or not message.name:
            return
        payload = (
            json.dumps(
                {
                    "type": "command",
                    "names": list(message.name),
                    "positions": [float(value) for value in message.position],
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        with self._socket_lock:
            connection = self._socket
        if connection is None:
            return
        try:
            connection.sendall(payload)
        except OSError:
            with self._socket_lock:
                if self._socket is connection:
                    self._socket = None

    def _send_heartbeat(self):
        with self._socket_lock:
            connection = self._socket
        if connection is None:
            return
        try:
            connection.sendall(b'{"type":"heartbeat"}\n')
        except OSError:
            with self._socket_lock:
                if self._socket is connection:
                    self._socket = None

    def destroy_node(self):
        self._stop.set()
        with self._socket_lock:
            connection = self._socket
            self._socket = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        self._thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IsaacTcpBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
