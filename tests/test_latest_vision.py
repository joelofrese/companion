import threading
import time
import unittest

from control.tracking import TrackEstimate
from vision.latest import LatestVisionPipeline


class _BlockingVision:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def process(self, frame, timestamp_s):
        self.started.set()
        self.release.wait(timeout=1.0)
        return TrackEstimate(10.0, 20.0, 0.0, 0.0, 10.0, 20.0, target_height_px=30.0)


class LatestVisionTests(unittest.TestCase):
    def test_reactive_reads_do_not_wait_for_inference_and_age_results(self):
        vision = _BlockingVision()
        latest = LatestVisionPipeline(vision)
        try:
            self.assertIsNone(latest.process("frame", 1.0))
            self.assertTrue(vision.started.wait(timeout=1.0))
            self.assertIsNone(latest.process(None, 1.1))
            vision.release.set()
            deadline = time.monotonic() + 1.0
            estimate = None
            while estimate is None and time.monotonic() < deadline:
                estimate = latest.process(None, 1.2)
                time.sleep(0.01)
            self.assertIsNotNone(estimate)
            self.assertAlmostEqual(estimate.age_s, 0.2)
        finally:
            latest.close()

    def test_inference_error_clears_previous_estimate(self):
        class FailingVision:
            def __init__(self):
                self.started = threading.Event()

            def process(self, frame, timestamp_s):
                self.started.set()
                raise RuntimeError("model failed")

        vision = FailingVision()
        latest = LatestVisionPipeline(vision)
        try:
            latest.process("frame", 1.0)
            self.assertTrue(vision.started.wait(timeout=1.0))
            self.assertIsNone(latest.process(None, 1.1))
        finally:
            latest.close()


if __name__ == "__main__":
    unittest.main()
