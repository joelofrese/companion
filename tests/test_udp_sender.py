import socket
import unittest

from control.command_packet import CommandPacket
from control.udp_sender import UdpCommandSender
from control.velocity import VelocityCommand


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def sendto(self, payload, destination):
        self.sent.append((payload, destination))

    def close(self):
        self.closed = True


class UdpCommandSenderTests(unittest.TestCase):
    def test_sends_versioned_sequential_packets(self):
        socket_instance = FakeSocket()
        sender = UdpCommandSender(
            "drone.local",
            5001,
            socket_factory=lambda *args: socket_instance,
        )
        sender.send(VelocityCommand(north_m_s=0.3))
        sender.send(VelocityCommand(east_m_s=-0.1))

        self.assertEqual(socket_instance.sent[0][1], ("drone.local", 5001))
        self.assertEqual(CommandPacket.decode(socket_instance.sent[0][0]).sequence, 0)
        self.assertEqual(CommandPacket.decode(socket_instance.sent[1][0]).sequence, 1)
        sender.close()
        self.assertTrue(socket_instance.closed)

    def test_invalid_destination_is_rejected(self):
        with self.assertRaises(ValueError):
            UdpCommandSender("", 5001)
        with self.assertRaises(ValueError):
            UdpCommandSender("drone.local", 0)
        with self.assertRaises(ValueError):
            UdpCommandSender("drone.local", 65536)

    def test_send_starts_socket_lazily(self):
        socket_instance = FakeSocket()
        created = []

        def create_socket(*args):
            created.append(args)
            return socket_instance

        sender = UdpCommandSender("127.0.0.1", socket_factory=create_socket)
        sender.start()
        sender.start()
        self.assertEqual(created, [(socket.AF_INET, socket.SOCK_DGRAM)])


if __name__ == "__main__":
    unittest.main()
