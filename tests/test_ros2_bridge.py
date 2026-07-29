import math
import time
import unittest

from control.udp_sender import UdpCommandSender
from control.velocity import VelocityCommand
from onboard.ros2_bridge import LatestDistanceSensor, Ros2SafetyBridge


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FailingPublisher(FakePublisher):
    def publish(self, message):
        raise RuntimeError("publisher failed")


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


class FailingNode(FakeNode):
    def create_publisher(self, message_type, topic, qos):
        publisher = FailingPublisher()
        self.publishers[topic] = publisher
        self.publisher_calls.append((message_type, topic, qos))
        return publisher


class FakeHeartbeat:
    pass


class FakeSetpoint:
    pass


class FakeDistance:
    def __init__(self, current_distance, min_distance=None, max_distance=None):
        self.current_distance = current_distance
        self.min_distance = min_distance
        self.max_distance = max_distance


class Ros2DistanceSensorTests(unittest.TestCase):
    def test_distance_expires_without_a_fresh_message(self):
        now = [10.0]
        sensor = LatestDistanceSensor(clock=lambda: now[0], timeout_s=0.15)
        self.assertTrue(math.isnan(sensor.read()))
        sensor.update(FakeDistance(2.0))
        self.assertEqual(sensor.read(), 2.0)
        now[0] = 10.151
        self.assertTrue(math.isnan(sensor.read()))

    def test_non_positive_timeout_is_rejected(self):
        for value in (0.0, float("nan"), True, "slow"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    LatestDistanceSensor(timeout_s=value)

    def test_out_of_range_or_malformed_readings_become_unsafe(self):
        sensor = LatestDistanceSensor(clock=lambda: 10.0)
        sensor.update(FakeDistance(0.1, min_distance=0.2, max_distance=4.0))
        self.assertTrue(math.isnan(sensor.read()))
        sensor.update(FakeDistance(5.0, min_distance=0.2, max_distance=4.0))
        self.assertTrue(math.isnan(sensor.read()))
        sensor.update(FakeDistance(True, min_distance=0.2, max_distance=4.0))
        self.assertTrue(math.isnan(sensor.read()))
        sensor.update(FakeDistance(-1.0))
        self.assertTrue(math.isnan(sensor.read()))


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
        sender = None
        try:
            bridge.start()
            sender = UdpCommandSender("127.0.0.1", bridge.port)
            self.assertTrue(all(call[3] == "px4-qos" for call in node.subscriptions))
            self.assertTrue(all(call[2] == "px4-qos" for call in node.publisher_calls))
            self.assertEqual(set(node.publishers), {
                "/fmu/in/offboard_control_mode",
                "/fmu/in/trajectory_setpoint",
            })
            sender.send(VelocityCommand(north_m_s=0.3))
            setpoints = node.publishers["/fmu/in/trajectory_setpoint"].messages
            deadline = time.monotonic() + 1.0
            while len(setpoints) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(setpoints)
            self.assertEqual(setpoints[-1].velocity, [0.0, 0.0, 0.0])

            node.subscriptions[0][2](FakeDistance(2.0))
            sender.send(VelocityCommand(north_m_s=0.3))
            deadline = time.monotonic() + 1.0
            while setpoints[-1].velocity != [0.3, 0.0, 0.0] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(setpoints[-1].velocity, [0.3, 0.0, 0.0])
            bridge.close()
            self.assertEqual(setpoints[-1].velocity, [0.0, 0.0, 0.0])
        finally:
            bridge.close()
            if sender is not None:
                sender.close()

    def test_forwarder_failure_is_observable_after_bridge_close(self):
        bridge = Ros2SafetyBridge(
            FailingNode(),
            FakeHeartbeat,
            FakeSetpoint,
            FakeDistance,
            qos_profile="px4-qos",
            bind_host="127.0.0.1",
            command_port=0,
            tick_period_s=0.01,
        )
        try:
            bridge.start()
            deadline = time.monotonic() + 1.0
            while bridge.error is None and time.monotonic() < deadline:
                time.sleep(0.01)
            bridge.close()
            self.assertIsInstance(bridge.error, RuntimeError)
        finally:
            bridge.close()

if __name__ == "__main__":
    unittest.main()
