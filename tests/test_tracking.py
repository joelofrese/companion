import unittest

from control.tracking import Detection, PersonTracker


class PersonTrackerTests(unittest.TestCase):
    def test_first_detection_initializes_position(self):
        estimate = PersonTracker().update(Detection(100.0, 50.0, 1.0))
        self.assertEqual(estimate.x_px, 100.0)
        self.assertEqual(estimate.y_px, 50.0)
        self.assertEqual(estimate.predicted_x_px, 100.0)

    def test_prediction_leads_a_moving_person(self):
        tracker = PersonTracker(prediction_horizon_s=0.3)
        tracker.update(Detection(0.0, 20.0, 0.0))
        estimate = tracker.update(Detection(100.0, 20.0, 1.0))
        self.assertGreater(estimate.vx_px_s, 0.0)
        self.assertGreater(estimate.predicted_x_px, estimate.x_px)

    def test_filter_reduces_single_measurement_jump(self):
        tracker = PersonTracker()
        tracker.update(Detection(0.0, 0.0, 0.0))
        estimate = tracker.update(Detection(100.0, 0.0, 0.1))
        self.assertLess(estimate.x_px, 100.0)

    def test_out_of_order_detection_is_rejected(self):
        tracker = PersonTracker()
        tracker.update(Detection(0.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            tracker.update(Detection(1.0, 0.0, 1.0))

    def test_negative_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            PersonTracker(prediction_horizon_s=-0.1)

    def test_non_finite_tracker_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            PersonTracker(prediction_horizon_s=float("nan"))
        with self.assertRaises(ValueError):
            PersonTracker(max_prediction_age_s=float("inf"))

    def test_malformed_detection_is_rejected_before_state_update(self):
        tracker = PersonTracker()
        with self.assertRaises(ValueError):
            tracker.update(Detection(float("nan"), 20.0, 0.0))
        with self.assertRaises(ValueError):
            tracker.update(Detection(10.0, 20.0, True))
        with self.assertRaises(ValueError):
            tracker.update(Detection(10.0, 20.0, 0.0, height_px=-1.0))
        self.assertIsNone(tracker.predict(0.0))

    def test_non_finite_prediction_timestamp_is_rejected(self):
        tracker = PersonTracker()
        tracker.update(Detection(10.0, 20.0, 0.0))
        with self.assertRaises(ValueError):
            tracker.predict(float("nan"))

    def test_prediction_bridges_a_short_detection_gap(self):
        tracker = PersonTracker(max_prediction_age_s=0.5)
        tracker.update(Detection(0.0, 20.0, 0.0))
        tracker.update(Detection(100.0, 20.0, 1.0))
        estimate = tracker.predict(1.2)
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.age_s, 0.2)
        self.assertGreater(estimate.predicted_x_px, estimate.x_px)

    def test_prediction_expires_a_stale_track(self):
        tracker = PersonTracker(max_prediction_age_s=0.5)
        tracker.update(Detection(10.0, 20.0, 0.0))
        self.assertIsNone(tracker.predict(0.6))


if __name__ == "__main__":
    unittest.main()
