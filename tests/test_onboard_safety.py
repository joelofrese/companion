import unittest

from control.velocity import VelocityCommand
from onboard.safety import OnboardSafetyEnvelope


class OnboardSafetyEnvelopeTests(unittest.TestCase):
    def test_no_command_is_zero(self):
        envelope = OnboardSafetyEnvelope()
        self.assertEqual(envelope.tick(1.0), VelocityCommand())

    def test_fresh_command_passes_through(self):
        envelope = OnboardSafetyEnvelope()
        command = VelocityCommand(north_m_s=0.3)
        envelope.receive(1.0, command)
        self.assertEqual(envelope.tick(1.1), command)

    def test_stale_command_expires_to_zero(self):
        envelope = OnboardSafetyEnvelope(command_timeout_s=0.15)
        envelope.receive(1.0, VelocityCommand(north_m_s=0.3))
        self.assertEqual(envelope.tick(1.151), VelocityCommand())

    def test_local_obstacle_overrides_fresh_command(self):
        envelope = OnboardSafetyEnvelope()
        envelope.receive(1.0, VelocityCommand(north_m_s=0.3))
        self.assertEqual(envelope.tick(1.1, obstacle_distance_m=0.5), VelocityCommand(north_m_s=-0.2))

    def test_out_of_order_commands_and_ticks_are_rejected(self):
        envelope = OnboardSafetyEnvelope()
        envelope.receive(2.0, VelocityCommand())
        with self.assertRaises(ValueError):
            envelope.receive(1.0, VelocityCommand())
        envelope.tick(2.1)
        with self.assertRaises(ValueError):
            envelope.tick(2.0)


if __name__ == "__main__":
    unittest.main()
