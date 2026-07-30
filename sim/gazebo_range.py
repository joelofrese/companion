"""Read a forward range sensor from Gazebo."""

import json
import math
import os
import queue
import shutil
import subprocess
import threading
from typing import Optional

from sim.offboard_control import DistanceMessage
from vision.video_stream import close_subprocess


class GazeboRangefinder:
    """Keep the newest forward Gazebo lidar reading."""

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
            raise RuntimeError("gz is required for lidar simulation")
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
                minimum = _finite_number(message["rangeMin"])
                maximum = _finite_number(message["rangeMax"])
                ranges = message["ranges"]
                values = [
                    value
                    for value in (_number_or_nan(item) for item in ranges)
                    if math.isfinite(value)
                ]
                distance = min(values) if values else math.nan
                sample = DistanceMessage(distance, minimum, maximum)
            except (KeyError, TypeError, ValueError):
                continue
            try:
                self._samples.put_nowait(sample)
            except queue.Full:
                self._samples.get_nowait()
                self._samples.put_nowait(sample)
        if not self._closed:
            self._failed = True

    def latest(self) -> Optional[DistanceMessage]:
        """Return the newest reading, or none when no new reading is ready."""

        if self._failed:
            raise RuntimeError("Gazebo lidar stream ended")
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


def _finite_number(value) -> float:
    if isinstance(value, bool):
        raise ValueError("range limit must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("range limit must be finite")
    return number


def _number_or_nan(value) -> float:
    if isinstance(value, bool):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan
