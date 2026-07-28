import tempfile
import unittest

from sim.video_loopback import image_sender_command
from vision.video_stream import H264StreamConfig


class VideoLoopbackTests(unittest.TestCase):
    def test_image_sender_uses_real_jpeg_and_configured_rtp(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            command = image_sender_command(
                H264StreamConfig(port=5012, width=640, height=480, framerate=15),
                image.name,
            )

        self.assertIn("multifilesrc", command)
        self.assertIn(f"location={image.name}", command)
        self.assertIn("videoscale", command)
        self.assertIn("port=5012", command)

    def test_image_sender_rejects_missing_image(self):
        with self.assertRaises(ValueError):
            image_sender_command(H264StreamConfig(), "/missing/person.jpg")


if __name__ == "__main__":
    unittest.main()
