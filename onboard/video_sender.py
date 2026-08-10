"""Send the CM5 camera to the companion brain."""

import argparse
import subprocess
import time
from dataclasses import replace

from vision.video_stream import H264StreamConfig, close_subprocess


class GStreamerH264Sender:
    """Run the CM5 camera pipeline."""

    def __init__(
        self,
        destination_host: str,
        config: H264StreamConfig = H264StreamConfig(),
    ):
        if not destination_host.strip():
            raise ValueError("destination host must not be empty")
        self.destination_host = destination_host
        self.config = config
        self._process = None

    def command(self):
        """Return the camera command."""

        return self.config.sender_command(self.destination_host)

    def start(self):
        """Start the sender."""

        if self._process is None:
            self._process = subprocess.Popen(
                self.command(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    @property
    def running(self):
        """Return whether the camera process is running."""

        return self._process is not None and self._process.poll() is None

    def close(self):
        """Stop the sender."""

        if self._process is None:
            return
        close_subprocess(self._process)
        self._process = None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stream the CM5 camera over RTP/H.264")
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
            if not sender.running:
                raise RuntimeError("camera streaming pipeline exited")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()


if __name__ == "__main__":
    main()
