import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from sim import offboard_video
from vision.video_stream import H264StreamConfig


class OffboardVideoTests(unittest.TestCase):
    def test_run_accepts_injected_vision_and_sender(self):
        async def scenario():
            injected_vision = object()
            with patch.object(offboard_video.GStreamerH264Receiver, "start"), \
                    patch.object(offboard_video.GStreamerH264Receiver, "close"), \
                    patch.object(offboard_video.subprocess, "Popen") as popen, \
                    patch.object(offboard_video, "close_subprocess"), \
                    patch.object(offboard_video.offboard, "run", new_callable=AsyncMock) as flight:
                await offboard_video.run(
                    config=H264StreamConfig(port=5013),
                    vision=injected_vision,
                    sender_command=["sender"],
                )
                popen.assert_called_once()
                self.assertIs(flight.await_args.kwargs["vision"], injected_vision)
                self.assertEqual(flight.await_args.kwargs["runtime"].controller.follower.config.frame_width_px, 640.0)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
