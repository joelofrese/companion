"""Receive the drone's H.264 video stream on the Mac."""

import asyncio
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def close_subprocess(process):
    """Stop a media process."""

    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@dataclass(frozen=True)
class H264StreamConfig:
    port: int = 5000
    width: int = 640
    height: int = 480
    framerate: int = 30

    def __post_init__(self):
        if not all(_positive_int(value) for value in (self.port, self.width, self.height, self.framerate)):
            raise ValueError("stream dimensions, port, and framerate must be positive")

    def sender_command(self, destination_host: str) -> Sequence[str]:
        """Return the CM5 camera command."""

        if not destination_host.strip():
            raise ValueError("destination host must not be empty")
        return (
            "gst-launch-1.0",
            "-q",
            "libcamerasrc",
            "!",
            f"video/x-raw,width={self.width},height={self.height},framerate={self.framerate}/1",
            "!",
            "videoconvert",
            "!",
            "x264enc",
            "tune=zerolatency",
            "speed-preset=ultrafast",
            "bitrate=1500",
            f"key-int-max={self.framerate}",
            "!",
            "rtph264pay",
            "config-interval=1",
            "pt=96",
            "!",
            "udpsink",
            f"host={destination_host}",
            f"port={self.port}",
        )


class GStreamerH264Receiver:
    """Decode RTP/H.264 into BGR frames."""

    def __init__(
        self,
        config: H264StreamConfig = H264StreamConfig(),
    ):
        self.config = config
        self._process = None

    def command(self) -> Sequence[str]:
        """Return the GStreamer receive command."""

        config = self.config
        return (
            "gst-launch-1.0",
            "-q",
            "udpsrc",
            f"port={config.port}",
            "caps=application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000",
            "!",
            "rtph264depay",
            "!",
            "h264parse",
            "!",
            "avdec_h264",
            "!",
            "videoconvert",
            "!",
            (
                "video/x-raw,format=BGR,"
                f"width={config.width},height={config.height},"
                f"framerate={config.framerate}/1"
            ),
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        )

    def _start(self):
        self._process = subprocess.Popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def start(self):
        """Start the decoder if needed."""

        if self._process is None:
            self._start()

    def read(self):
        """Return a timestamp and frame, or none at end of stream."""

        self.start()
        frame_bytes = self._read_exact(self._process.stdout, self.config.width * self.config.height * 3)
        if frame_bytes is None:
            return None

        import numpy as np

        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            (self.config.height, self.config.width, 3)
        )
        return time.monotonic(), frame

    @staticmethod
    def _read_exact(stream, size: int):
        chunks = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self):
        if self._process is None:
            return
        close_subprocess(self._process)
        self._process = None


class AsyncLatestFrameReader:
    """Read frames without blocking the control loop."""

    def __init__(self, receiver: GStreamerH264Receiver):
        self.receiver = receiver
        self._pending: Optional[asyncio.Task] = None
        self._ended = False
        self._last_timestamp_s: Optional[float] = None

    def _sample(self, timestamp_s: float, frame: Any):
        timestamp_s = max(timestamp_s, time.monotonic())
        if self._last_timestamp_s is not None:
            timestamp_s = max(timestamp_s, self._last_timestamp_s + 1e-6)
        self._last_timestamp_s = timestamp_s
        return timestamp_s, frame

    async def read(self):
        """Return a frame, or an empty sample when none is ready."""

        if self._ended:
            return self._sample(time.monotonic(), None)
        if self._pending is None:
            self._pending = asyncio.create_task(asyncio.to_thread(self.receiver.read))
        if not self._pending.done():
            return self._sample(time.monotonic(), None)

        result = await self._pending
        self._pending = None
        if result is None:
            self._ended = True
            return self._sample(time.monotonic(), None)
        return self._sample(*result)
