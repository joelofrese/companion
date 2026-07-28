"""Run the offboard scenario with decoded RTP video frames in its control loop."""

import asyncio
import subprocess

from control.following import FollowConfig, VisualFollower
from control.runtime import CompanionRuntime
from control.state_machine import ReactiveController
from sim import offboard
from sim.offboard_control import VideoDemoVision
from sim.video_loopback import synthetic_sender_command
from vision.video_stream import (
    AsyncLatestFrameReader,
    GStreamerH264Receiver,
    H264StreamConfig,
    close_subprocess,
)


async def run(
    config=H264StreamConfig(port=5011, width=64, height=48, framerate=30),
    vision=None,
    sender_command=None,
):
    receiver = GStreamerH264Receiver(config)
    frame_reader = AsyncLatestFrameReader(receiver)
    sender = None
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

    try:
        receiver.start()
        sender = subprocess.Popen(
            sender_command or synthetic_sender_command(config),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("RTP video loopback started.")
        await offboard.run(
            vision=vision or VideoDemoVision(),
            runtime=runtime,
            frame_reader=frame_reader.read,
        )
    finally:
        receiver.close()
        if sender is not None:
            close_subprocess(sender)
        close = getattr(vision, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    asyncio.run(run())
