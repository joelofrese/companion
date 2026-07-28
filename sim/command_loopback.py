"""Verify the Mac-to-CM5 command service over local UDP."""

import asyncio

from control.udp_sender import UdpCommandSender
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService


class RecordingForwarder:
    def __init__(self):
        self.commands = asyncio.Queue()

    async def send(self, command):
        await self.commands.put(command)


async def _next_command(forwarder):
    return await asyncio.wait_for(forwarder.commands.get(), timeout=1.0)


async def run():
    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    sender = None
    forwarder = RecordingForwarder()
    stop_event = asyncio.Event()
    obstacle_distance_m = None

    def obstacle_distance():
        return obstacle_distance_m

    service = SafetyCommandService(receiver, forwarder, obstacle_distance=obstacle_distance)
    service_task = None
    try:
        service.start()
        service_task = asyncio.create_task(service.run(stop_event))
        sender = UdpCommandSender("127.0.0.1", receiver.port)
        sender.send(VelocityCommand(north_m_s=0.3))
        command = VelocityCommand()
        while command != VelocityCommand(north_m_s=0.3):
            command = await _next_command(forwarder)

        obstacle_distance_m = 0.5
        obstacle_command = await _next_command(forwarder)
        while obstacle_command != VelocityCommand(north_m_s=-0.2):
            obstacle_command = await _next_command(forwarder)

        stop_event.set()
        await service_task
        if await _next_command(forwarder) != VelocityCommand():
            raise RuntimeError("command service did not send zero on shutdown")
        print(f"Fresh command={command}; obstacle command={obstacle_command}; shutdown=zero")
    finally:
        if sender is not None:
            sender.close()
        if service_task is not None and not service_task.done():
            stop_event.set()
            await service_task


if __name__ == "__main__":
    asyncio.run(run())
