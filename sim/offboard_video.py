"""Run the offboard scenario with decoded RTP video frames in its control loop."""

import asyncio
import subprocess

from control.following import FollowConfig, VisualFollower
from control.runtime import CompanionRuntime
from control.state_machine import ReactiveController
from sim import offboard
from sim.offboard_control import VideoDemoVision
from sim.video_loopback import synthetic_sender_command
from vision.video_stream import GStreamerH264Receiver, H264StreamConfig


async def run():
    config = H264StreamConfig(port=5011, width=64, height=48, framerate=30)
    receiver = GStreamerH264Receiver(config)
    sender = None
    runtime = CompanionRuntime(
        ReactiveController(
            VisualFollower(
                FollowConfig(
                    frame_width_px=config.width,
                    desired_target_height_px=config.height / 8.0,
                )
            )
        )
    )

    async def read_frame():
        return await asyncio.to_thread(receiver.read)

    try:
        receiver.start()
        sender = subprocess.Popen(
            synthetic_sender_command(config),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("RTP video loopback started.")
        await offboard.run(
            vision=VideoDemoVision(),
            runtime=runtime,
            frame_reader=read_frame,
        )
    finally:
        receiver.close()
        if sender is not None:
            sender.terminate()
            sender.wait(timeout=5)


if __name__ == "__main__":
    asyncio.run(run())
