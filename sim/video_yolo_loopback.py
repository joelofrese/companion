"""Verify real JPEG -> RTP/H.264 -> YOLO -> tracking -> reactive control."""

import subprocess
import sys

from control.following import FollowConfig, VisualFollower
from control.runtime import CompanionRuntime
from control.state_machine import ReactiveController, State
from sim.video_loopback import image_sender_command
from vision.person_detector import YoloPersonDetector
from vision.pipeline import PersonVisionPipeline
from vision.video_stream import GStreamerH264Receiver, H264StreamConfig, close_subprocess


def run(image_path: str):
    config = H264StreamConfig(port=5012, width=640, height=480, framerate=15)
    receiver = GStreamerH264Receiver(config)
    sender = None
    try:
        receiver.start()
        sender = subprocess.Popen(
            image_sender_command(config, image_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        sample = receiver.read()
        if sample is None:
            error = sender.stderr.read().decode(errors="replace") if sender.stderr else ""
            raise RuntimeError(f"RTP image stream ended before a frame arrived: {error}")
        timestamp_s, frame = sample
        pipeline = PersonVisionPipeline(YoloPersonDetector(model_path="yolov8n.pt"))
        estimate = pipeline.process(frame, timestamp_s)
        if estimate is None:
            raise RuntimeError("YOLO did not detect a person in the RTP-decoded image")

        runtime = CompanionRuntime(
            ReactiveController(
                VisualFollower(
                    FollowConfig(
                        frame_width_px=config.width,
                        desired_target_height_px=config.height / 4.0,
                    )
                )
            )
        )
        runtime.set_intent(State.FOLLOWING)
        runtime.update_target(estimate, timestamp_s)
        command = runtime.command(timestamp_s)
        print(
            f"Received frame shape={frame.shape}; detected person "
            f"height={estimate.target_height_px:.1f}px; command={command}"
        )
    finally:
        receiver.close()
        if sender is not None:
            close_subprocess(sender)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m sim.video_yolo_loopback IMAGE_PATH")
    run(sys.argv[1])
