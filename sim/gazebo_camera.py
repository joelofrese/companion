"""Read raw camera frames from a running Gazebo world."""

import base64
import json
import os
import queue
import shutil
import subprocess
import threading
from typing import Optional

import numpy as np


class GazeboCamera:
    """Keep the newest Gazebo camera frame available to the brain."""

    def __init__(self, topic: str):
        self.topic = topic
        self._frames = queue.Queue(maxsize=1)
        self._process = None
        self._thread = None
        self._closed = False
        self._failed = False

    def start(self):
        gz = shutil.which("gz")
        if gz is None:
            raise RuntimeError("gz is required for camera simulation")
        environment = os.environ.copy()
        environment["GZ_IP"] = "127.0.0.1"
        self._process = subprocess.Popen(
            [gz, "topic", "-e", "--json-output", "-t", self.topic],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=environment,
        )
        self._closed = False
        self._failed = False
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            try:
                message = json.loads(line)
                frame = base64.b64decode(message["data"])
                width = int(message["width"])
                height = int(message["height"])
                step = int(message["step"])
                image = np.frombuffer(frame, dtype=np.uint8).reshape(height, step)
                image = image[:, : width * 3]
                image = image.reshape(height, width, 3)[:, :, ::-1]
                try:
                    self._frames.put_nowait(image)
                except queue.Full:
                    self._frames.get_nowait()
                    self._frames.put_nowait(image)
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
        if not self._closed:
            self._failed = True

    def latest(self) -> Optional[np.ndarray]:
        """Return the newest BGR frame, if Gazebo has produced one."""

        if self._failed:
            raise RuntimeError("Gazebo camera stream ended")
        try:
            return self._frames.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        if self._process is None:
            return
        self._closed = True
        self._process.terminate()
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None
