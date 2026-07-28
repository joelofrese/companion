import unittest

from vision.video_stream import GStreamerH264Receiver, H264StreamConfig


class FakeStdout:
    def __init__(self, payload, chunk_size=5):
        self.payload = payload
        self.chunk_size = chunk_size

    def read(self, size):
        chunk = self.payload[: min(size, self.chunk_size)]
        self.payload = self.payload[len(chunk):]
        return chunk


class FakeProcess:
    def __init__(self, payload):
        self.stdout = FakeStdout(payload)
        self.terminated = False
        self.waited = False

    def terminate(self):
        self.terminated = True

    def wait(self):
        self.waited = True


class GStreamerH264ReceiverTests(unittest.TestCase):
    def test_command_contains_rtp_h264_decode_chain(self):
        receiver = GStreamerH264Receiver(H264StreamConfig(port=6000, width=4, height=3, framerate=25))
        command = receiver.command()
        self.assertIn("port=6000", command)
        self.assertIn("rtph264depay", command)
        self.assertIn("avdec_h264", command)
        self.assertIn("video/x-raw,format=BGR,width=4,height=3,framerate=25/1", command)

    def test_read_handles_partial_pipe_reads_and_shapes_bgr_frame(self):
        config = H264StreamConfig(width=4, height=3)
        process = FakeProcess(bytes(range(config.width * config.height * 3)))
        receiver = GStreamerH264Receiver(config, process_factory=lambda *args, **kwargs: process)

        timestamp, frame = receiver.read()

        self.assertIsInstance(timestamp, float)
        self.assertEqual(frame.shape, (3, 4, 3))
        self.assertEqual(int(frame[0, 0, 0]), 0)
        self.assertEqual(int(frame[2, 3, 2]), 35)

    def test_eof_returns_none_and_close_releases_process(self):
        process = FakeProcess(b"")
        receiver = GStreamerH264Receiver(
            H264StreamConfig(width=2, height=2),
            process_factory=lambda *args, **kwargs: process,
        )
        self.assertIsNone(receiver.read())
        receiver.close()
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            GStreamerH264Receiver(H264StreamConfig(port=0))


if __name__ == "__main__":
    unittest.main()
