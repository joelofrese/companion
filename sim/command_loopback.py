"""Verify the Mac-to-CM5 command link over local UDP."""

import asyncio
import math
import time

from control.udp_sender import UdpCommandSender
from control.udp_control import UdpControlService
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService


class FixedCommandControl:
    """Provide one normal Mac command for the transport loopback."""

    def __init__(self):
        self.obstacle_distances = []

    def tick(self, frame, timestamp_s, intent=None, obstacle_distance_m=None):
        self.obstacle_distances.append(obstacle_distance_m)
        return VelocityCommand(north_m_s=0.25)


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

    def cm5_obstacle_distance():
        return 2.0 if time.monotonic() - cm5_started_at < 0.12 else 0.5

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
    control = FixedCommandControl()
    try:
        cm5_service.start()
        cm5_started_at = time.monotonic()
        cm5_task = asyncio.create_task(cm5_service.run(cm5_stop_event))
        sender = UdpCommandSender("127.0.0.1", receiver.port)
        mac_task = asyncio.create_task(
            UdpControlService(
                control,
                sender,
                frame_reader,
                obstacle_provider=sender.obstacle_distance,
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
        if not any(
            distance is not None
            and not (isinstance(distance, float) and math.isnan(distance))
            and distance > 1.0
            for distance in control.obstacle_distances
        ):
            raise RuntimeError(
                f"Mac did not receive a clear CM5 distance: {control.obstacle_distances}"
            )
        if not any(
            distance is not None
            and not (isinstance(distance, float) and math.isnan(distance))
            and distance < 0.6
            for distance in control.obstacle_distances
        ):
            raise RuntimeError(
                f"Mac did not receive the CM5 obstacle distance: {control.obstacle_distances}"
            )
        cm5_stop_event.set()
        await cm5_task
        shutdown_command = await _next_command(forwarder)
        if shutdown_command != VelocityCommand():
            raise RuntimeError("command service did not send zero on shutdown")
        print(
            "Mac control service=verified; CM5 telemetry=verified; "
            "obstacle command=verified; shutdown=zero"
        )
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
