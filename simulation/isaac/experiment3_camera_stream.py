"""Stream the rendered experiment-three top camera to the ROS host over TCP."""

from __future__ import annotations

import socket
import struct
import threading
import time


def normalize_rgb_frame(data, width: int, height: int):
    """Normalize Replicator RGB/RGBA data to an HxWx3 uint8 array."""
    import numpy as np

    array = np.asarray(data)
    if array.ndim == 3:
        if array.shape[0] != height or array.shape[1] != width or array.shape[2] < 3:
            raise ValueError(f"unexpected camera frame shape: {array.shape}")
        return array[:, :, :3].astype(np.uint8, copy=False)
    if array.ndim == 1:
        pixel_count = width * height
        if array.size == pixel_count * 4:
            return array.reshape(height, width, 4)[:, :, :3].astype(np.uint8, copy=False)
        if array.size == pixel_count * 3:
            return array.reshape(height, width, 3).astype(np.uint8, copy=False)
    raise ValueError(f"unexpected camera frame shape: {array.shape}")


class IsaacCameraTcpStream:
    """Publish the latest Replicator RGB frame as a length-prefixed JPEG."""

    def __init__(
        self,
        camera_path: str,
        width: int = 848,
        height: int = 480,
        bind_host: str = "0.0.0.0",
        port: int = 8766,
        publish_rate_hz: float = 10.0,
    ) -> None:
        import cv2
        import omni.replicator.core as rep

        if width <= 0 or height <= 0 or publish_rate_hz <= 0.0:
            raise ValueError("camera dimensions and publish rate must be positive")
        self._cv2 = cv2
        self._width = int(width)
        self._height = int(height)
        self._period = 1.0 / float(publish_rate_hz)
        self._next_frame_time = 0.0
        self._rep = rep
        self._render_product = rep.create.render_product(
            camera_path, (int(width), int(height)), name="experiment3_top_camera"
        )
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotator.attach([self._render_product])
        self._client = None
        self._client_lock = threading.Lock()
        self._stop = threading.Event()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((bind_host, int(port)))
        self._server.listen(1)
        self._server.settimeout(0.5)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        print(f"Isaac camera stream listening on {bind_host}:{port}", flush=True)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                client, address = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._client_lock:
                previous = self._client
                self._client = client
            if previous is not None:
                try:
                    previous.close()
                except OSError:
                    pass
            print(f"Isaac camera stream connected: {address}", flush=True)

    def update(self) -> None:
        now = time.monotonic()
        if now < self._next_frame_time:
            return
        self._next_frame_time = now + self._period
        data = self._annotator.get_data()
        if data is None:
            return
        try:
            frame = normalize_rgb_frame(data, self._width, self._height)
        except ValueError as exc:
            print(f"Isaac camera frame skipped: {exc}", flush=True)
            return
        success, encoded = self._cv2.imencode(
            ".jpg", self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR),
            [self._cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        if not success:
            return
        payload = encoded.tobytes()
        packet = struct.pack("!I", len(payload)) + payload
        with self._client_lock:
            client = self._client
        if client is None:
            return
        try:
            client.sendall(packet)
        except OSError:
            with self._client_lock:
                if self._client is client:
                    self._client = None
            try:
                client.close()
            except OSError:
                pass

    def close(self) -> None:
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
        try:
            self._annotator.detach([self._render_product])
        except Exception:
            pass
