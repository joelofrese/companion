import unittest

from control.tracking import Detection
from vision.pipeline import PersonVisionPipeline


class FakeDetector:
    def __init__(self, detections):
        self.detections = iter(detections)
        self.calls = []

    def detect(self, frame, timestamp_s):
        self.calls.append((frame, timestamp_s))
        return next(self.detections)


class PersonVisionPipelineTests(unittest.TestCase):
    def test_composes_detection_with_tracking_prediction(self):
        detector = FakeDetector([
            Detection(10.0, 20.0, 0.0),
            Detection(30.0, 20.0, 1.0),
        ])
        pipeline = PersonVisionPipeline(detector)

        pipeline.process("first", 0.0)
        estimate = pipeline.process("second", 1.0)

        self.assertGreater(estimate.predicted_x_px, estimate.x_px)
        self.assertEqual(detector.calls, [("first", 0.0), ("second", 1.0)])

    def test_missing_detection_produces_no_target(self):
        detector = FakeDetector([None])
        pipeline = PersonVisionPipeline(detector)
        self.assertIsNone(pipeline.process("empty", 4.0))

    def test_missing_detection_bridges_a_short_gap(self):
        detector = FakeDetector([
            Detection(10.0, 20.0, 0.0),
            Detection(30.0, 20.0, 1.0),
            None,
        ])
        pipeline = PersonVisionPipeline(detector)
        pipeline.process("first", 0.0)
        pipeline.process("second", 1.0)
        estimate = pipeline.process("gap", 1.2)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.age_s, 0.2)

    def test_missing_frame_returns_prediction_without_calling_detector(self):
        detector = FakeDetector([Detection(10.0, 20.0, 0.0)])
        pipeline = PersonVisionPipeline(detector)
        pipeline.process("first", 0.0)
        self.assertIsNotNone(pipeline.process(None, 0.1))
        self.assertEqual(detector.calls, [("first", 0.0)])

    def test_missing_detection_expires_a_stale_target(self):
        detector = FakeDetector([
            Detection(10.0, 20.0, 0.0),
            None,
        ])
        pipeline = PersonVisionPipeline(detector)
        pipeline.process("first", 0.0)
        self.assertIsNone(pipeline.process("stale", 0.6))

    def test_tracker_errors_are_not_hidden(self):
        detector = FakeDetector([
            Detection(10.0, 20.0, 2.0),
            Detection(20.0, 20.0, 1.0),
        ])
        pipeline = PersonVisionPipeline(detector)
        pipeline.process("first", 2.0)
        with self.assertRaises(ValueError):
            pipeline.process("late", 1.0)


if __name__ == "__main__":
    unittest.main()
