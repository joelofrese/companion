"""Mac-side receiver for the drone's H.264 RTP video stream."""

import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class H264StreamConfig:
    port: int = 5000
    width: int = 640
    height: int = 480
    framerate: int = 30


class GStreamerH264Receiver:
    """Decode RTP/H.264 into BGR frames using a GStreamer subprocess."""

    def __init__(
        self,
        config: H264StreamConfig = H264StreamConfig(),
        process_factory: Optional[Callable[..., object]] = None,
    ):
        if config.port <= 0 or config.width <= 0 or config.height <= 0 or config.framerate <= 0:
            raise ValueError("stream dimensions, port, and framerate must be positive")
        self.config = config
        self._process_factory = process_factory or subprocess.Popen
        self._process = None

    def command(self) -> Sequence[str]:
        """Return the concrete GStreamer command used by the receiver."""

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
            "avdec_h264",
            "!",
            "videoconvert",
            "!",
            f"video/x-raw,format=BGR,width={config.width},height={config.height},framerate={config.framerate}/1",
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        )

    def _start(self):
        self._process = self._process_factory(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def read(self):
        """Return ``(monotonic_timestamp_s, BGR_frame)`` or None at stream EOF."""

        if self._process is None:
            self._start()
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
        self._process.terminate()
        self._process.wait()
        self._process = None
