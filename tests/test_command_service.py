import asyncio
import unittest

from control.velocity import VelocityCommand
from onboard.command_service import SafetyCommandService


class FakeReceiver:
    def __init__(self):
        self.started = False
        self.closed = False
        self.obstacles = []

    def start(self):
        self.started = True

    def poll(self, obstacle_distance_m=None):
        self.obstacles.append(obstacle_distance_m)
        return VelocityCommand(north_m_s=0.3)

    def close(self):
        self.closed = True


class FakeForwarder:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.commands = []

    async def send(self, command):
        self.commands.append(command)
        if len(self.commands) == 1:
            self.stop_event.set()


class FailingForwarder:
    async def send(self, command):
        raise RuntimeError("forwarder unavailable")


class SafetyCommandServiceTests(unittest.TestCase):
    def test_lifecycle_forwards_then_stops_with_zero(self):
        async def scenario():
            stop_event = asyncio.Event()
            receiver = FakeReceiver()
            forwarder = FakeForwarder(stop_event)
            service = SafetyCommandService(
                receiver,
                forwarder,
                obstacle_distance=lambda: 0.8,
            )

            await service.run(stop_event)
            return receiver, forwarder

        receiver, forwarder = asyncio.run(scenario())
        self.assertTrue(receiver.started)
        self.assertTrue(receiver.closed)
        self.assertEqual(receiver.obstacles, [0.8])
        self.assertEqual(forwarder.commands, [VelocityCommand(north_m_s=0.3), VelocityCommand()])

    def test_start_binds_before_run(self):
        receiver = FakeReceiver()
        service = SafetyCommandService(receiver, object())
        service.start()
        self.assertTrue(receiver.started)

    def test_tick_period_must_be_positive(self):
        with self.assertRaises(ValueError):
            SafetyCommandService(FakeReceiver(), object(), tick_period_s=0.0)

    def test_receiver_closes_when_forwarder_fails(self):
        async def scenario():
            receiver = FakeReceiver()
            service = SafetyCommandService(receiver, FailingForwarder())
            with self.assertRaises(RuntimeError):
                await service.run(asyncio.Event())
            return receiver

        receiver = asyncio.run(scenario())
        self.assertTrue(receiver.closed)


if __name__ == "__main__":
    unittest.main()
