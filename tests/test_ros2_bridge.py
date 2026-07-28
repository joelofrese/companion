import socket
import time
import unittest

from control.command_packet import CommandPacket
from control.velocity import VelocityCommand
from onboard.ros2_bridge import Ros2SafetyBridge


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeClockTime:
    nanoseconds = 2_000_000


class FakeClock:
    def now(self):
        return FakeClockTime()


class FakeNode:
    def __init__(self):
        self.publishers = {}
        self.publisher_calls = []
        self.subscriptions = []

    def create_publisher(self, message_type, topic, qos):
        publisher = FakePublisher()
        self.publishers[topic] = publisher
        self.publisher_calls.append((message_type, topic, qos))
        return publisher

    def create_subscription(self, message_type, topic, callback, qos):
        self.subscriptions.append((message_type, topic, callback, qos))

    def get_clock(self):
        return FakeClock()


class FakeHeartbeat:
    pass


class FakeSetpoint:
    pass


class FakeDistance:
    def __init__(self, current_distance):
        self.current_distance = current_distance


class Ros2SafetyBridgeTests(unittest.TestCase):
    def test_missing_sensor_stays_zero_then_forwards_after_valid_distance(self):
        node = FakeNode()
        bridge = Ros2SafetyBridge(
            node,
            FakeHeartbeat,
            FakeSetpoint,
            FakeDistance,
            qos_profile="px4-qos",
            bind_host="127.0.0.1",
            command_port=0,
            tick_period_s=0.01,
        )
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            bridge.start()
            self.assertTrue(all(call[3] == "px4-qos" for call in node.subscriptions))
            self.assertTrue(all(call[2] == "px4-qos" for call in node.publisher_calls))
            self.assertEqual(set(node.publishers), {
                "/fmu/in/offboard_control_mode",
                "/fmu/in/trajectory_setpoint",
            })
            sender.sendto(
                CommandPacket(1, VelocityCommand(north_m_s=0.3)).encode(),
                ("127.0.0.1", bridge.port),
            )
            setpoints = node.publishers["/fmu/in/trajectory_setpoint"].messages
            deadline = time.monotonic() + 1.0
            while len(setpoints) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(setpoints)
            self.assertEqual(setpoints[-1].velocity, [0.0, 0.0, 0.0])

            node.subscriptions[0][2](FakeDistance(2.0))
            sender.sendto(
                CommandPacket(2, VelocityCommand(north_m_s=0.3)).encode(),
                ("127.0.0.1", bridge.port),
            )
            deadline = time.monotonic() + 1.0
            while setpoints[-1].velocity != [0.3, 0.0, 0.0] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(setpoints[-1].velocity, [0.3, 0.0, 0.0])
            bridge.close()
            self.assertEqual(setpoints[-1].velocity, [0.0, 0.0, 0.0])
        finally:
            bridge.close()
            sender.close()

if __name__ == "__main__":
    unittest.main()
