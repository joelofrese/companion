import unittest

from control.command_packet import CommandPacket
from control.velocity import VelocityCommand


class CommandPacketTests(unittest.TestCase):
    def test_round_trip_preserves_sequence_and_velocity(self):
        packet = CommandPacket(7, VelocityCommand(0.3, -0.1, 0.05, 12.0))
        self.assertEqual(CommandPacket.decode(packet.encode()), packet)

    def test_decode_rejects_bad_version_and_malformed_payload(self):
        with self.assertRaises(ValueError):
            CommandPacket.decode(b"not json")
        with self.assertRaises(ValueError):
            CommandPacket.decode(b'{"version":2,"sequence":1,"velocity":{}}')

    def test_decode_rejects_non_finite_velocity(self):
        payload = b'{"version":1,"sequence":1,"velocity":{"north_m_s":NaN,"east_m_s":0,"down_m_s":0,"yaw_deg":0}}'
        with self.assertRaises(ValueError):
            CommandPacket.decode(payload)

    def test_negative_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            CommandPacket(-1, VelocityCommand()).encode()

    def test_encode_rejects_non_finite_velocity(self):
        with self.assertRaises(ValueError):
            CommandPacket(1, VelocityCommand(north_m_s=float("inf"))).encode()

    def test_encode_rejects_boolean_velocity(self):
        with self.assertRaises(ValueError):
            CommandPacket(1, VelocityCommand(north_m_s=True)).encode()


if __name__ == "__main__":
    unittest.main()
