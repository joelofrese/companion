"""Read decoded JSON messages from one Gazebo topic."""

import json
import os
import queue
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
        self._messages = queue.Queue(maxsize=1)
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
            try:
                self._messages.put_nowait(message)
            except queue.Full:
                self._messages.get_nowait()
                self._messages.put_nowait(message)
        if not self._closed:
            self._failed = True

    def latest(self) -> Optional[Any]:
        """Return the newest decoded message, or none when not ready."""

        if self._failed:
            raise RuntimeError(f"Gazebo {self._name} stream ended")
        try:
            return self._messages.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        if self._process is None:
            return
        self._closed = True
        close_subprocess(self._process)
        self._process = None
