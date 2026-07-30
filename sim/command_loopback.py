"""Verify the Mac-to-CM5 command link over local UDP."""

import asyncio
import time

from control.runtime import CompanionRuntime
from control.reactive import State
from control.udp_sender import UdpCommandSender
from control.udp_control import UdpControlService
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from sim.offboard_control import DemoVision


class RecordingForwarder:
    def __init__(self):
        self.commands = asyncio.Queue()

    async def send(self, command):
        await self.commands.put(command)


async def _next_command(forwarder):
    return await asyncio.wait_for(forwarder.commands.get(), timeout=1.0)


async def run():
    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    forwarder = RecordingForwarder()
    cm5_stop_event = asyncio.Event()
    mac_stop_event = asyncio.Event()
    frame_count = 0
    cm5_started_at = None

    def mac_obstacle_distance():
        return 0.5 if frame_count >= 2 else 2.0

    def cm5_obstacle_distance():
        return 2.0 if time.monotonic() - cm5_started_at < 0.08 else 0.5

    async def frame_reader():
        nonlocal frame_count
        frame_count += 1
        await asyncio.sleep(0.05)
        if frame_count >= 3:
            mac_stop_event.set()
        return time.monotonic(), None

    cm5_service = SafetyCommandService(receiver, forwarder, obstacle_distance=cm5_obstacle_distance)
    cm5_task = None
    mac_task = None
    sender = None
    try:
        cm5_service.start()
        cm5_started_at = time.monotonic()
        cm5_task = asyncio.create_task(cm5_service.run(cm5_stop_event))
        sender = UdpCommandSender("127.0.0.1", receiver.port)
        mac_task = asyncio.create_task(
            UdpControlService(
                CompanionRuntime(DemoVision()),
                sender,
                frame_reader,
                intent_provider=lambda timestamp_s: State.FOLLOWING,
                obstacle_provider=lambda timestamp_s: mac_obstacle_distance(),
                tick_period_s=0.01,
            ).run(mac_stop_event)
        )
        await mac_task
        await asyncio.sleep(0.05)
        commands = []
        while not forwarder.commands.empty():
            commands.append(forwarder.commands.get_nowait())
        if VelocityCommand(north_m_s=0.25) not in commands:
            raise RuntimeError(f"Mac control service did not produce a fresh follow command: {commands}")
        if VelocityCommand(north_m_s=-0.2) not in commands:
            raise RuntimeError(f"Mac control service did not produce obstacle backoff: {commands}")
        cm5_stop_event.set()
        await cm5_task
        shutdown_command = await _next_command(forwarder)
        if shutdown_command != VelocityCommand():
            raise RuntimeError("command service did not send zero on shutdown")
        print("Mac control service=verified; obstacle command=verified; shutdown=zero")
    finally:
        if sender is not None:
            sender.close()
        if mac_task is not None and not mac_task.done():
            mac_stop_event.set()
            await mac_task
        if cm5_task is not None and not cm5_task.done():
            cm5_stop_event.set()
            await cm5_task


if __name__ == "__main__":
    asyncio.run(run())
