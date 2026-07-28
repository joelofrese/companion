"""CM5 command service that applies local safety before forwarding to PX4."""

import asyncio
from typing import Callable, Optional

from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver


class SafetyCommandService:
    """Run the CM5 receive, safety, and forwarding loop at a fixed rate."""

    def __init__(
        self,
        receiver: UdpSafetyReceiver,
        forwarder,
        tick_period_s: float = 0.02,
        obstacle_distance: Optional[Callable[[], Optional[float]]] = None,
    ):
        if tick_period_s <= 0.0:
            raise ValueError("tick period must be positive")
        self.receiver = receiver
        self.forwarder = forwarder
        self.tick_period_s = tick_period_s
        self.obstacle_distance = obstacle_distance or (lambda: None)

    def start(self):
        """Bind the receiver before the Mac is allowed to send packets."""

        self.receiver.start()

    async def run(self, stop_event: asyncio.Event):
        """Forward safe commands until requested to stop, then send zero once."""

        self.start()
        try:
            while not stop_event.is_set():
                command = self.receiver.poll(obstacle_distance_m=self.obstacle_distance())
                await self.forwarder.send(command)
                await asyncio.sleep(self.tick_period_s)
        finally:
            await self.forwarder.send(VelocityCommand())
            self.receiver.close()
