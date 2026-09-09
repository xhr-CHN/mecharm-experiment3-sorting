"""Publish a deterministic RGB8 overhead-camera frame for the local simulation graph."""

from __future__ import annotations

import json


def make_rgb_frame(width: int, height: int, rectangles=()) -> bytes:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    pixels = bytearray(width * height * 3)
    for left, top, right, bottom, color in rectangles:
        left = max(0, min(width, int(left)))
        right = max(0, min(width, int(right)))
        top = max(0, min(height, int(top)))
        bottom = max(0, min(height, int(bottom)))
        if right <= left or bottom <= top:
            continue
        red, green, blue = (int(value) for value in color)
        for y in range(top, bottom):
            row = (y * width + left) * 3
            for x in range(left, right):
                offset = row + (x - left) * 3
                pixels[offset : offset + 3] = bytes((red, green, blue))
    return bytes(pixels)


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class SimCameraNode(Node):
        def __init__(self) -> None:
            super().__init__("sim_camera_node")
            self.declare_parameter("image_topic", "/camera/image_raw")
            self.declare_parameter("frame_id", "top_camera")
            self.declare_parameter("width", 848)
            self.declare_parameter("height", 480)
            self.declare_parameter("publish_rate_hz", 5.0)
            self.declare_parameter("config_path", "")
            self.width = int(self.get_parameter("width").value)
            self.height = int(self.get_parameter("height").value)
            self.rectangles = self._read_rectangles()
            self.publisher = self.create_publisher(
                Image, str(self.get_parameter("image_topic").value), 10
            )
            self.timer = self.create_timer(
                1.0 / float(self.get_parameter("publish_rate_hz").value),
                self._publish,
            )

        def _read_rectangles(self):
            config_path = str(self.get_parameter("config_path").value)
            if not config_path:
                return ()
            import yaml

            with open(config_path, encoding="utf-8") as stream:
                document = yaml.safe_load(stream) or {}
            settings = document.get("sim_camera", {}).get("ros__parameters", {})
            raw = settings.get("rectangles_json", "[]")
            entries = json.loads(str(raw))
            return tuple(
                (
                    float(item["left"]),
                    float(item["top"]),
                    float(item["right"]),
                    float(item["bottom"]),
                    tuple(item["color"]),
                )
                for item in entries
            )

        def _publish(self) -> None:
            message = Image()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = str(self.get_parameter("frame_id").value)
            message.height = self.height
            message.width = self.width
            message.encoding = "rgb8"
            message.is_bigendian = False
            message.step = self.width * 3
            message.data = make_rgb_frame(self.width, self.height, self.rectangles)
            self.publisher.publish(message)

    rclpy.init(args=args)
    node = SimCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
