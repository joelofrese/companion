import unittest

from control.command_packet import CommandPacket
from control.velocity import VelocityCommand


class CommandPacketTests(unittest.TestCase):
    def test_round_trip_preserves_sequence_and_velocity(self):
        packet = CommandPacket(7, VelocityCommand(0.3, -0.1, 0.05, 12.0))
        self.assertEqual(CommandPacket.decode(packet.encode()), packet)

    def test_rejects_malformed_or_unsafe_values(self):
        with self.assertRaises(ValueError):
            CommandPacket.decode(b"not json")
        with self.assertRaises(ValueError):
            CommandPacket.decode(b'{"version":2,"sequence":1,"velocity":{}}')
        with self.assertRaises(ValueError):
            CommandPacket.decode(b'{"version":1,"sequence":1,"velocity":{"north_m_s":NaN,"east_m_s":0,"down_m_s":0,"yaw_deg":0}}')
        with self.assertRaises(ValueError):
            CommandPacket(-1, VelocityCommand()).encode()
        with self.assertRaises(ValueError):
            CommandPacket(1, VelocityCommand(north_m_s=float("inf"))).encode()
        with self.assertRaises(ValueError):
            CommandPacket(1, VelocityCommand(north_m_s=True)).encode()


if __name__ == "__main__":
    unittest.main()
