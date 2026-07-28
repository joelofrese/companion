"""Verify the local RTP/H.264 transport with a synthetic GStreamer camera."""

import subprocess

from vision.video_stream import GStreamerH264Receiver, H264StreamConfig


def synthetic_sender_command(config: H264StreamConfig):
    """Return a local GStreamer sender matching the production RTP format."""

    return [
        "gst-launch-1.0",
        "-q",
        "videotestsrc",
        "is-live=true",
        "!",
        f"video/x-raw,width={config.width},height={config.height},framerate={config.framerate}/1",
        "!",
        "videoconvert",
        "!",
        "x264enc",
        "tune=zerolatency",
        "speed-preset=ultrafast",
        "bitrate=1500",
        f"key-int-max={config.framerate}",
        "!",
        "rtph264pay",
        "config-interval=1",
        "pt=96",
        "!",
        "udpsink",
        "host=127.0.0.1",
        f"port={config.port}",
    ]


def run():
    config = H264StreamConfig(port=5010, width=64, height=48, framerate=30)
    receiver = GStreamerH264Receiver(config)
    sender = None
    try:
        receiver.start()
        sender = subprocess.Popen(
            synthetic_sender_command(config),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        result = receiver.read()
        if result is None:
            raise RuntimeError("video loopback ended before a frame arrived")
        timestamp_s, frame = result
        if frame.max() == 0:
            raise RuntimeError("video loopback produced an empty frame")
        print(f"Received BGR frame shape={frame.shape} at {timestamp_s:.3f}s")
    finally:
        receiver.close()
        if sender is not None:
            sender.terminate()
            sender.wait(timeout=5)


if __name__ == "__main__":
    run()
