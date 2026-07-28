"""Mac-side control service that streams reactive commands to the CM5."""

import asyncio
from typing import Awaitable, Callable, Optional, Protocol

from control.loop import CompanionControlLoop
from control.state_machine import State
from control.udp_sender import UdpCommandSender
from control.velocity import VelocityCommand


class FrameReader(Protocol):
    def __call__(self) -> Awaitable[Optional[tuple[float, object]]]:
        ...


class UdpControlService:
    """Run the Mac reactive heartbeat and send one explicit zero on shutdown."""

    def __init__(
        self,
        control_loop: CompanionControlLoop,
        sender: UdpCommandSender,
        frame_reader: FrameReader,
        intent_provider: Optional[Callable[[float], Optional[State]]] = None,
        obstacle_provider: Optional[Callable[[float], Optional[float]]] = None,
        tick_period_s: float = 0.05,
    ):
        if tick_period_s <= 0.0:
            raise ValueError("tick period must be positive")
        self.control_loop = control_loop
        self.sender = sender
        self.frame_reader = frame_reader
        self.intent_provider = intent_provider or (lambda timestamp_s: None)
        self.obstacle_provider = obstacle_provider or (lambda timestamp_s: None)
        self.tick_period_s = tick_period_s

    async def run(self, stop_event: asyncio.Event):
        """Run until stopped; input termination is an error and always fails safe."""

        self.sender.start()
        try:
            while not stop_event.is_set():
                sample = await self.frame_reader()
                if sample is None:
                    raise RuntimeError("video stream ended before control shutdown")
                timestamp_s, frame = sample
                command = self.control_loop.tick(
                    frame=frame,
                    timestamp_s=timestamp_s,
                    intent=self.intent_provider(timestamp_s),
                    obstacle_distance_m=self.obstacle_provider(timestamp_s),
                )
                self.sender.send(command)
                await asyncio.sleep(self.tick_period_s)
        finally:
            try:
                self.sender.send(VelocityCommand())
            finally:
                self.sender.close()
