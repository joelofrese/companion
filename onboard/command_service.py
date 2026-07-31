"""Run the CM5 safety loop."""

import asyncio
import math
from numbers import Real
from typing import Callable, Optional

from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver


class SafetyCommandService:
    """Read, check, and forward commands at a fixed rate."""

    def __init__(
        self,
        receiver: UdpSafetyReceiver,
        forwarder,
        tick_period_s: float = 0.02,
        obstacle_distance: Optional[Callable[[], Optional[float]]] = None,
        velocity_provider: Optional[
            Callable[[], tuple[Optional[float], Optional[float], Optional[float]]]
        ] = None,
    ):
        if (
            isinstance(tick_period_s, bool)
            or not isinstance(tick_period_s, Real)
            or not math.isfinite(tick_period_s)
            or tick_period_s <= 0.0
        ):
            raise ValueError("tick period must be positive")
        self.receiver = receiver
        self.forwarder = forwarder
        self.tick_period_s = tick_period_s
        self.obstacle_distance = obstacle_distance or (lambda: None)
        self.velocity_provider = velocity_provider or (lambda: (None, None, None))

    def start(self):
        """Start the receiver."""

        self.receiver.start()

    async def run(self, stop_event: asyncio.Event):
        """Forward safe commands until stopped, then send zero."""

        try:
            while not stop_event.is_set():
                obstacle_distance_m = self.obstacle_distance()
                command = self.receiver.poll(obstacle_distance_m=obstacle_distance_m)
                self.receiver.send_telemetry(
                    obstacle_distance_m,
                    *self.velocity_provider(),
                )
                await self.forwarder.send(command)
                await asyncio.sleep(self.tick_period_s)
        finally:
            try:
                await self.forwarder.send(VelocityCommand())
            finally:
                self.receiver.close()
