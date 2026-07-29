"""Build the local RTP/H.264 image sender."""

from pathlib import Path

from vision.video_stream import H264StreamConfig


def image_sender_command(config: H264StreamConfig, image_path: str):
    """Return a command that repeats one JPEG frame."""

    if not Path(image_path).is_file():
        raise ValueError(f"image path does not exist: {image_path}")
    return [
        "gst-launch-1.0",
        "-q",
        "multifilesrc",
        f"location={image_path}",
        "loop=true",
        f"caps=image/jpeg,framerate={config.framerate}/1",
        "!",
        "jpegdec",
        "!",
        "videoscale",
        "!",
        f"video/x-raw,width={config.width},height={config.height}",
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
