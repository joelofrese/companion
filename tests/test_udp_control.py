import asyncio
import unittest

from control.udp_control import UdpControlService
from control.velocity import VelocityCommand


class FakeLoop:
    def tick(self, **kwargs):
        return VelocityCommand(north_m_s=0.3)


class FakeSender:
    def __init__(self):
        self.commands = []
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def send(self, command):
        self.commands.append(command)

    def close(self):
        self.closed = True


class UdpControlServiceTests(unittest.TestCase):
    def test_streams_commands_and_sends_shutdown_zero(self):
        sender = FakeSender()
        samples = iter([(1.0, "frame")])
        stop_event = None

        async def frame_reader():
            sample = next(samples)
            stop_event.set()
            return sample

        async def scenario():
            nonlocal stop_event
            stop_event = asyncio.Event()
            await UdpControlService(
                FakeLoop(), sender, frame_reader, tick_period_s=0.01
            ).run(stop_event)

        asyncio.run(scenario())
        self.assertTrue(sender.started)
        self.assertEqual(sender.commands, [VelocityCommand(north_m_s=0.3), VelocityCommand()])
        self.assertTrue(sender.closed)

    def test_ended_video_fails_safe_and_closes_sender(self):
        sender = FakeSender()

        async def frame_reader():
            return None

        async def scenario():
            with self.assertRaises(RuntimeError):
                await UdpControlService(FakeLoop(), sender, frame_reader).run(asyncio.Event())

        asyncio.run(scenario())
        self.assertEqual(sender.commands, [VelocityCommand()])
        self.assertTrue(sender.closed)

    def test_stalled_video_fails_safe_and_closes_sender(self):
        sender = FakeSender()
        samples = iter([(1.0, None), (1.6, None)])

        async def frame_reader():
            return next(samples)

        async def scenario():
            with self.assertRaisesRegex(RuntimeError, "video stream stalled"):
                await UdpControlService(
                    FakeLoop(),
                    sender,
                    frame_reader,
                    tick_period_s=0.001,
                    frame_timeout_s=0.5,
                ).run(asyncio.Event())

        asyncio.run(scenario())
        self.assertEqual(sender.commands[-1], VelocityCommand())
        self.assertTrue(sender.closed)

    def test_invalid_tick_period_is_rejected(self):
        with self.assertRaises(ValueError):
            UdpControlService(FakeLoop(), FakeSender(), lambda: None, tick_period_s=0.0)
        with self.assertRaises(ValueError):
            UdpControlService(FakeLoop(), FakeSender(), lambda: None, frame_timeout_s=0.0)


if __name__ == "__main__":
    unittest.main()
