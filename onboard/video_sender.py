"""Managed CM5 camera sender for the Mac RTP/H.264 receiver."""

import argparse
import subprocess
import time
from dataclasses import replace
from typing import Callable, Optional

from vision.video_stream import H264StreamConfig, close_subprocess


class GStreamerH264Sender:
    """Run the low-latency libcamera-to-RTP pipeline on the CM5."""

    def __init__(
        self,
        destination_host: str,
        config: H264StreamConfig = H264StreamConfig(),
        process_factory: Optional[Callable[..., object]] = None,
    ):
        if not destination_host.strip():
            raise ValueError("destination host must not be empty")
        self.destination_host = destination_host
        self.config = config
        self._process_factory = process_factory or subprocess.Popen
        self._process = None

    def command(self):
        """Return the concrete CM5 camera sender command."""

        return self.config.sender_command(self.destination_host)

    def start(self):
        """Start the sender once; repeated starts are harmless."""

        if self._process is None:
            self._process = self._process_factory(
                self.command(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def close(self):
        """Stop the sender and release its child process."""

        if self._process is None:
            return
        close_subprocess(self._process)
        self._process = None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stream the CM5 camera to the Mac over RTP/H.264")
    parser.add_argument("destination_host")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--framerate", type=int, default=30)
    args = parser.parse_args(argv)
    config = replace(
        H264StreamConfig(),
        port=args.port,
        width=args.width,
        height=args.height,
        framerate=args.framerate,
    )
    sender = GStreamerH264Sender(args.destination_host, config)
    try:
        sender.start()
        print(f"Streaming camera to {args.destination_host}:{args.port}.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()


if __name__ == "__main__":
    main()
