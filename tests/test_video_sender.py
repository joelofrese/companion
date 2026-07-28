import unittest

from onboard.video_sender import GStreamerH264Sender
from vision.video_stream import H264StreamConfig


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.waited = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True

    def kill(self):
        self.killed = True


class GStreamerH264SenderTests(unittest.TestCase):
    def test_command_uses_camera_rtp_contract(self):
        sender = GStreamerH264Sender(
            "192.168.1.20",
            H264StreamConfig(port=6000, width=320, height=240, framerate=20),
        )
        command = sender.command()
        self.assertIn("libcamerasrc", command)
        self.assertIn("video/x-raw,width=320,height=240,framerate=20/1", command)
        self.assertIn("host=192.168.1.20", command)
        self.assertIn("port=6000", command)

    def test_start_is_idempotent_and_close_releases_process(self):
        process = FakeProcess()
        sender = GStreamerH264Sender(
            "127.0.0.1",
            process_factory=lambda *args, **kwargs: process,
        )
        sender.start()
        sender.start()
        sender.close()
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    def test_empty_destination_is_rejected(self):
        with self.assertRaises(ValueError):
            GStreamerH264Sender(" ")


if __name__ == "__main__":
    unittest.main()
