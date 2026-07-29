import unittest

from control.command_packet import CommandPacket
from control.velocity import VelocityCommand
from onboard.safety import OnboardSafetyEnvelope


class OnboardSafetyEnvelopeTests(unittest.TestCase):
    def test_invalid_command_timeout_is_rejected(self):
        for value in (0.0, float("nan"), True, "fast"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    OnboardSafetyEnvelope(command_timeout_s=value)

    def test_absent_or_stale_commands_are_zero(self):
        envelope = OnboardSafetyEnvelope()
        self.assertEqual(envelope.tick(1.0), VelocityCommand())
        envelope = OnboardSafetyEnvelope(command_timeout_s=0.15)
        envelope.receive(1.0, VelocityCommand(north_m_s=0.3))
        self.assertEqual(envelope.tick(1.151), VelocityCommand())

    def test_local_obstacle_overrides_fresh_command(self):
        envelope = OnboardSafetyEnvelope()
        envelope.receive(1.0, VelocityCommand(north_m_s=0.3))
        self.assertEqual(envelope.tick(1.1, obstacle_distance_m=0.5), VelocityCommand(north_m_s=-0.2))

    def test_invalid_obstacle_reading_fails_safe_to_zero(self):
        envelope = OnboardSafetyEnvelope()
        envelope.receive(1.0, VelocityCommand(north_m_s=0.3))
        self.assertEqual(envelope.tick(1.1, obstacle_distance_m=float("nan")), VelocityCommand())
        self.assertEqual(envelope.tick(1.2, obstacle_distance_m=float("inf")), VelocityCommand())
        self.assertEqual(envelope.tick(1.3, obstacle_distance_m="unknown"), VelocityCommand())
        self.assertEqual(envelope.tick(1.4, obstacle_distance_m=True), VelocityCommand())
        self.assertEqual(envelope.tick(1.5, obstacle_distance_m=False), VelocityCommand())
        self.assertEqual(envelope.tick(1.6, obstacle_distance_m=-1.0), VelocityCommand())

    def test_out_of_bounds_or_reordered_commands_fail_safe(self):
        envelope = OnboardSafetyEnvelope()
        envelope.receive(1.0, VelocityCommand(north_m_s=0.5, east_m_s=-0.5, down_m_s=0.3))
        self.assertEqual(
            envelope.tick(1.1),
            VelocityCommand(north_m_s=0.5, east_m_s=-0.5, down_m_s=0.3),
        )

        envelope.receive(1.2, VelocityCommand(north_m_s=0.51))
        self.assertEqual(envelope.tick(1.3), VelocityCommand())

        envelope.receive(1.4, VelocityCommand(down_m_s=-0.31))
        self.assertEqual(envelope.tick(1.5), VelocityCommand())
        envelope = OnboardSafetyEnvelope()
        first = CommandPacket(2, VelocityCommand(north_m_s=0.3)).encode()
        old = CommandPacket(1, VelocityCommand(north_m_s=-0.3)).encode()
        self.assertTrue(envelope.receive_packet(1.0, first))
        self.assertFalse(envelope.receive_packet(1.1, old))
        self.assertEqual(envelope.tick(1.151), VelocityCommand())


if __name__ == "__main__":
    unittest.main()
