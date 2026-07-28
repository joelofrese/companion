import asyncio
import subprocess
import unittest

from sim.video_loopback import synthetic_sender_command
from vision.video_stream import (
    AsyncLatestFrameReader,
    GStreamerH264Receiver,
    H264StreamConfig,
    close_subprocess,
)


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
        self.killed = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True

    def kill(self):
        self.killed = True


class StuckProcess(FakeProcess):
    def wait(self, timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("gst-launch-1.0", timeout)
        self.waited = True


class FakeReceiver:
    def __init__(self, result):
        self.result = result
        self.read_count = 0

    def read(self):
        self.read_count += 1
        return self.result


class GStreamerH264ReceiverTests(unittest.TestCase):
    def test_stuck_media_process_is_killed_after_timeout(self):
        process = StuckProcess(b"")
        close_subprocess(process)
        self.assertTrue(process.killed)
        self.assertTrue(process.waited)

    def test_command_contains_rtp_h264_decode_chain(self):
        receiver = GStreamerH264Receiver(H264StreamConfig(port=6000, width=4, height=3, framerate=25))
        command = receiver.command()
        self.assertIn("port=6000", command)
        self.assertIn("rtph264depay", command)
        self.assertIn("h264parse", command)
        self.assertIn("avdec_h264", command)
        self.assertIn("video/x-raw,format=BGR,width=4,height=3,framerate=25/1", command)

    def test_sender_command_contains_low_latency_camera_to_rtp_chain(self):
        config = H264StreamConfig(port=6000, width=4, height=3, framerate=25)
        command = config.sender_command("192.168.1.20")
        self.assertIn("libcamerasrc", command)
        self.assertIn("x264enc", command)
        self.assertIn("tune=zerolatency", command)
        self.assertIn("rtph264pay", command)
        self.assertIn("host=192.168.1.20", command)
        self.assertIn("port=6000", command)

    def test_sender_rejects_empty_destination(self):
        with self.assertRaises(ValueError):
            H264StreamConfig().sender_command(" ")

    def test_synthetic_sender_matches_receiver_format(self):
        config = H264StreamConfig(port=6000, width=4, height=3, framerate=25)
        command = synthetic_sender_command(config)
        self.assertIn("videotestsrc", command)
        self.assertIn("video/x-raw,width=4,height=3,framerate=25/1", command)
        self.assertIn("port=6000", command)

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
            H264StreamConfig(port=0)
        with self.assertRaises(ValueError):
            H264StreamConfig(width=0)

    def test_async_reader_does_not_block_while_frame_is_pending(self):
        async def verify():
            receiver = FakeReceiver((1.0, "frame"))
            reader = AsyncLatestFrameReader(receiver)
            first = await reader.read()
            self.assertIsNone(first[1])
            second = first
            for _ in range(20):
                await asyncio.sleep(0.001)
                second = await reader.read()
                if second[1] is not None:
                    break
            self.assertEqual(second[1], "frame")
            self.assertGreater(second[0], first[0])

        asyncio.run(verify())

    def test_async_reader_keeps_capture_timestamps_monotonic(self):
        async def verify():
            receiver = FakeReceiver((1.0, "frame"))
            reader = AsyncLatestFrameReader(receiver)
            samples = [await reader.read()]
            for _ in range(20):
                await asyncio.sleep(0.001)
                samples.append(await reader.read())
                if samples[-1][1] is not None:
                    break
            self.assertGreater(samples[-1][0], samples[-2][0])

        asyncio.run(verify())


if __name__ == "__main__":
    unittest.main()
