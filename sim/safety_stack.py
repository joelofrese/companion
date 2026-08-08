"""Start the simulated Mac-to-CM5 safety path."""

import asyncio
from typing import Callable, Optional

from control.udp_sender import UdpCommandSender
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from sim.flight import RecordingForwarder
from sim.mavsdk_forwarder import MavsdkVelocityForwarder
from sim.offboard_control import SETPOINT_PERIOD_S


class SimulatedSafetyStack:
    """Own the shared simulated CM5 service and command link."""

    def __init__(
        self,
        drone,
        heading_provider: Callable[[], Optional[float]],
        obstacle_distance: Callable[[], Optional[float]],
        velocity_provider: Callable[
            [], tuple[Optional[float], Optional[float], Optional[float]]
        ],
        tick_period_s: float = SETPOINT_PERIOD_S,
    ):
        self.receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
        self.forwarder = RecordingForwarder(
            MavsdkVelocityForwarder(drone, heading_provider)
        )
        self.obstacle_distance = obstacle_distance
        self.velocity_provider = velocity_provider
        self.tick_period_s = tick_period_s
        self._stop_event = asyncio.Event()
        self._task = None
        self.sender = None

    def start(self):
        """Start CM5 safety and return the Mac command sender."""

        service = SafetyCommandService(
            self.receiver,
            self.forwarder,
            tick_period_s=self.tick_period_s,
            obstacle_distance=self.obstacle_distance,
            velocity_provider=self.velocity_provider,
        )
        service.start()
        self._task = asyncio.create_task(service.run(self._stop_event))
        self.sender = UdpCommandSender("127.0.0.1", self.receiver.port)
        return self.sender

    async def stop(self):
        """Stop CM5 safety after it has sent its final zero command."""

        if self._task is None:
            return
        self._stop_event.set()
        try:
            await self._task
        finally:
            self._task = None
            if self.sender is not None:
                self.sender.close()
                self.sender = None
