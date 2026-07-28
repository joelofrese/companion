"""Run PX4 SITL with the production RTP, YOLO, and tracking path."""

import asyncio
import sys

from sim import offboard_video
from sim.video_loopback import image_sender_command
from vision.person_detector import YoloPersonDetector
from vision.pipeline import PersonVisionPipeline
from vision.video_stream import H264StreamConfig
from vision.latest import LatestVisionPipeline


async def run(image_path: str):
    config = H264StreamConfig(port=5013, width=640, height=480, framerate=15)
    vision = LatestVisionPipeline(
        PersonVisionPipeline(YoloPersonDetector(model_path="yolov8n.pt"))
    )
    await offboard_video.run(
        config=config,
        vision=vision,
        sender_command=image_sender_command(config, image_path),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m sim.offboard_video_yolo IMAGE_PATH")
    asyncio.run(run(sys.argv[1]))
