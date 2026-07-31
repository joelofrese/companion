"""Read a Gazebo depth image as one forward distance."""

import base64
import json
import math
import os
import queue
import shutil
import subprocess
import threading
from typing import Optional

import numpy as np

from sim.offboard_control import DistanceMessage
from vision.video_stream import close_subprocess


MIN_DEPTH_M = 0.2
MAX_DEPTH_M = 19.1


class GazeboDepthRangefinder:
    """Keep the nearest valid distance in the center of a depth image."""

    def __init__(self, topic: str):
        self.topic = topic
        self._samples = queue.Queue(maxsize=1)
        self._process = None
        self._thread = None
        self._closed = False
        self._failed = False

    def start(self):
        gz = shutil.which("gz")
        if gz is None:
            raise RuntimeError("gz is required for depth simulation")
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
                width = int(message["width"])
                height = int(message["height"])
                step = int(message["step"])
                if step % 4:
                    raise ValueError("depth step is not a multiple of four")
                values = np.frombuffer(
                    base64.b64decode(message["data"]),
                    dtype="<f4",
                    count=(step // 4) * height,
                ).reshape(height, step // 4)[:, :width]
                center = values[height // 4:3 * height // 4, width // 4:3 * width // 4]
                valid = center[np.isfinite(center)]
                valid = valid[(valid >= MIN_DEPTH_M) & (valid <= MAX_DEPTH_M)]
                distance = float(np.min(valid)) if valid.size else math.nan
                sample = DistanceMessage(distance, MIN_DEPTH_M, MAX_DEPTH_M)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            try:
                self._samples.put_nowait(sample)
            except queue.Full:
                self._samples.get_nowait()
                self._samples.put_nowait(sample)
        if not self._closed:
            self._failed = True

    def latest(self) -> Optional[DistanceMessage]:
        """Return the newest depth reading, or none when not ready."""

        if self._failed:
            raise RuntimeError("Gazebo depth stream ended")
        try:
            return self._samples.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        if self._process is None:
            return
        self._closed = True
        close_subprocess(self._process)
        self._process = None
