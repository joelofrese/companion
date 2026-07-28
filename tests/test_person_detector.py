import unittest

from vision.person_detector import YoloPersonDetector


class FakeBox:
    def __init__(self, class_id, confidence, coordinates):
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [coordinates]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def __call__(self, frame, verbose=False):
        self.calls.append((frame, verbose))
        return self.results


class YoloPersonDetectorTests(unittest.TestCase):
    def test_returns_highest_confidence_person_center(self):
        model = FakeModel([FakeResult([
            FakeBox(0, 0.7, [10, 20, 30, 60]),
            FakeBox(0, 0.9, [100, 40, 140, 100]),
            FakeBox(1, 0.99, [0, 0, 200, 200]),
        ])])
        detector = YoloPersonDetector(model=model)

        detection = detector.detect("frame", timestamp_s=2.5)

        self.assertEqual((detection.x_px, detection.y_px), (120.0, 70.0))
        self.assertEqual(detection.confidence, 0.9)
        self.assertEqual((detection.width_px, detection.height_px), (40.0, 60.0))
        self.assertEqual(model.calls, [("frame", False)])

    def test_ignores_below_threshold_and_non_person_boxes(self):
        model = FakeModel([FakeResult([
            FakeBox(1, 0.99, [0, 0, 10, 10]),
            FakeBox(0, 0.49, [10, 10, 30, 30]),
        ])])
        detector = YoloPersonDetector(model=model)
        self.assertIsNone(detector.detect(None, timestamp_s=0.0))

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            YoloPersonDetector(model=FakeModel([]), confidence_threshold=1.1)
        with self.assertRaises(ValueError):
            YoloPersonDetector(model=FakeModel([]), confidence_threshold=float("nan"))
        with self.assertRaises(ValueError):
            YoloPersonDetector(model=FakeModel([]), confidence_threshold=True)

    def test_malformed_model_boxes_are_ignored(self):
        model = FakeModel([FakeResult([
            FakeBox(0, float("nan"), [0, 0, 10, 10]),
            FakeBox(0, 0.9, [10, 10, 5, 20]),
            FakeBox(0, 0.8, [0, 0, float("inf"), 10]),
        ])])
        self.assertIsNone(YoloPersonDetector(model=model).detect("frame", 1.0))


if __name__ == "__main__":
    unittest.main()
