"""Receive the Isaac top-camera JPEG stream and publish sensor_msgs/Image."""

from __future__ import annotations

import socket
import struct
import threading


def _receive_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def main(args=None) -> None:
    import cv2
    import numpy as np
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class VirtualCameraNode(Node):
        def __init__(self) -> None:
            super().__init__("virtual_camera_node")
            self.declare_parameter("host", "127.0.0.1")
            self.declare_parameter("port", 8766)
            self.declare_parameter("image_topic", "/camera/image_raw")
            self.declare_parameter("frame_id", "top_camera")
            self.host = str(self.get_parameter("host").value)
            self.port = int(self.get_parameter("port").value)
            self.publisher = self.create_publisher(
                Image, str(self.get_parameter("image_topic").value), 10
            )
            self.bridge = CvBridge()
            self._lock = threading.Lock()
            self._latest_jpeg: bytes | None = None
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()
            self.timer = self.create_timer(0.05, self._publish_latest)

        def _receive_loop(self) -> None:
            while not self._stop.is_set():
                connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Isaac rendering currently produces frames at about 1-2 Hz;
                # do not treat a slow frame as a broken TCP connection.
                connection.settimeout(10.0)
                try:
                    connection.connect((self.host, self.port))
                    self.get_logger().info(
                        f"connected to Isaac camera stream {self.host}:{self.port}"
                    )
                    while not self._stop.is_set():
                        header = _receive_exact(connection, 4)
                        if header is None:
                            break
                        payload_size = struct.unpack("!I", header)[0]
                        if payload_size <= 0 or payload_size > 20 * 1024 * 1024:
                            raise ValueError("invalid camera JPEG payload size")
                        payload = _receive_exact(connection, payload_size)
                        if payload is None:
                            break
                        with self._lock:
                            self._latest_jpeg = payload
                except (OSError, ValueError) as exc:
                    self.get_logger().warning(f"camera stream unavailable: {exc}")
                finally:
                    connection.close()
                self._stop.wait(1.0)

        def _publish_latest(self) -> None:
            with self._lock:
                payload = self._latest_jpeg
                self._latest_jpeg = None
            if payload is None:
                return
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return
            message = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = str(self.get_parameter("frame_id").value)
            self.publisher.publish(message)

        def destroy_node(self):
            self._stop.set()
            self._thread.join(timeout=1.0)
            super().destroy_node()

    rclpy.init(args=args)
    node = VirtualCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
