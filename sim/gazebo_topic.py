"""Read decoded JSON messages from one Gazebo topic."""

import json
import os
import shutil
import subprocess
import threading
from typing import Any, Callable, Optional

from vision.video_stream import close_subprocess


class GazeboTopicReader:
    """Keep the newest decoded message from a Gazebo topic."""

    def __init__(self, topic: str, decode: Callable[[dict], Any], name: str):
        self.topic = topic
        self._decode = decode
        self._name = name
        self._message = None
        self._message_lock = threading.Lock()
        self._process = None
        self._closed = False
        self._failed = False

    def start(self):
        gz = shutil.which("gz")
        if gz is None:
            raise RuntimeError(f"gz is required for {self._name} simulation")
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
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            try:
                message = self._decode(json.loads(line))
            except (KeyError, TypeError, ValueError):
                continue
            with self._message_lock:
                self._message = message
        if not self._closed:
            self._failed = True

    def latest(self) -> Optional[Any]:
        """Return the newest decoded message, or none when not ready."""

        if self._failed:
            raise RuntimeError(f"Gazebo {self._name} stream ended")
        with self._message_lock:
            message = self._message
            self._message = None
        return message

    def close(self):
        if self._process is None:
            return
        self._closed = True
        close_subprocess(self._process)
        self._process = None


class GazeboPoseAnimator:
    """Move one visual model through fixed poses in a background worker."""

    def __init__(
        self,
        world_name: str,
        model_name: str,
        poses,
        interval_s: float = 4.0,
    ):
        gz = shutil.which("gz")
        if gz is None:
            raise RuntimeError("gz is required for moving-person simulation")
        if not poses or interval_s <= 0.0:
            raise ValueError("pose animation needs poses and a positive interval")
        self._gz = gz
        self._service = f"/world/{world_name}/set_pose"
        self._model_name = model_name
        self._poses = tuple(poses)
        self._interval_s = interval_s
        self._environment = os.environ.copy()
        self._environment["GZ_IP"] = "127.0.0.1"
        self._stop = threading.Event()
        self._thread = None
        self._error = None

    def start(self):
        """Start moving the model after the first interval."""

        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="gazebo-pose-animation",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        index = 0
        try:
            while not self._stop.wait(self._interval_s):
                self._set_pose(self._poses[index])
                index = (index + 1) % len(self._poses)
        except Exception as error:
            self._error = error

    def _set_pose(self, pose):
        x, y, z = pose
        request = (
            f'name: "{self._model_name}" '
            f"position: {{x: {x} y: {y} z: {z}}} "
            "orientation: {w: 1}"
        )
        result = subprocess.run(
            [
                self._gz,
                "service",
                "-s",
                self._service,
                "--reqtype",
                "gz.msgs.Pose",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                "2000",
                "--req",
                request,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3.0,
            env=self._environment,
            check=False,
        )
        if result.returncode != 0 or "data: true" not in result.stdout:
            raise RuntimeError("Gazebo could not move the visual model")

    def check(self):
        """Raise a worker error in the simulation loop."""

        if self._error is not None:
            raise RuntimeError("Gazebo pose animation failed") from self._error

    def close(self):
        """Stop the worker and report any failed pose update."""

        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._thread = None
        self.check()
