import unittest

from control.velocity import VelocityCommand
from control.watchdog import SetpointWatchdog


class SetpointWatchdogTests(unittest.TestCase):
    def test_healthy_ticks_pass_desired_command(self):
        watchdog = SetpointWatchdog(max_interval_s=0.15)
        desired = VelocityCommand(north_m_s=0.5)
        self.assertEqual(watchdog.emit(1.0, desired), desired)
        self.assertEqual(watchdog.emit(1.1, desired), desired)
        self.assertFalse(watchdog.tripped)

    def test_missed_deadline_latches_zero_velocity(self):
        watchdog = SetpointWatchdog(max_interval_s=0.15)
        desired = VelocityCommand(north_m_s=0.5)
        watchdog.emit(1.0, desired)
        self.assertEqual(watchdog.emit(1.2, desired), VelocityCommand())
        self.assertTrue(watchdog.tripped)
        self.assertEqual(watchdog.emit(1.25, desired), VelocityCommand())

    def test_non_monotonic_timestamps_are_rejected(self):
        watchdog = SetpointWatchdog()
        watchdog.emit(1.0, VelocityCommand())
        with self.assertRaises(ValueError):
            watchdog.emit(1.0, VelocityCommand())

    def test_invalid_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            SetpointWatchdog(max_interval_s=0.0)


if __name__ == "__main__":
    unittest.main()
