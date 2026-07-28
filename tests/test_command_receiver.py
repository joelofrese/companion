import unittest

from control.command_packet import CommandPacket
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver


class FakeSocket:
    def __init__(self):
        self.payloads = []
        self.closed = False

    def bind(self, address):
        self.address = address

    def getsockname(self):
        return self.address

    def setblocking(self, enabled):
        self.blocking = enabled

    def recvfrom(self, size):
        if not self.payloads:
            raise BlockingIOError
        return self.payloads.pop(0), ("127.0.0.1", 12345)

    def close(self):
        self.closed = True


class UdpSafetyReceiverTests(unittest.TestCase):
    def test_loopback_packet_reaches_safety_envelope(self):
        fake_socket = FakeSocket()
        receiver = UdpSafetyReceiver(
            bind_host="127.0.0.1",
            port=5001,
            socket_factory=lambda *args: fake_socket,
        )
        receiver.start()
        fake_socket.payloads.append(CommandPacket(1, VelocityCommand(north_m_s=0.3)).encode())
        self.assertEqual(receiver.poll(), VelocityCommand(north_m_s=0.3))
        receiver.close()
        self.assertTrue(fake_socket.closed)

    def test_local_obstacle_overrides_received_packet(self):
        fake_socket = FakeSocket()
        receiver = UdpSafetyReceiver(socket_factory=lambda *args: fake_socket)
        receiver.start()
        fake_socket.payloads.append(CommandPacket(1, VelocityCommand(north_m_s=0.3)).encode())
        self.assertEqual(receiver.poll(obstacle_distance_m=0.5), VelocityCommand(north_m_s=-0.2))
        receiver.close()

    def test_poll_requires_start(self):
        with self.assertRaises(RuntimeError):
            UdpSafetyReceiver().poll()

    def test_boolean_port_is_rejected(self):
        with self.assertRaises(ValueError):
            UdpSafetyReceiver(port=True)


if __name__ == "__main__":
    unittest.main()
