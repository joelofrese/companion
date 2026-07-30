"""Send Mac control commands to the CM5."""

import asyncio
import math
from numbers import Real
from typing import Any, Awaitable, Callable, Optional, Protocol

from control.reactive import State
from control.udp_sender import UdpCommandSender
from control.velocity import VelocityCommand


class FrameReader(Protocol):
    def __call__(self) -> Awaitable[Optional[tuple[float, object]]]:
        ...


class ControlLoop(Protocol):
    def tick(
        self,
        frame: Any,
        timestamp_s: float,
        intent: Optional[object] = None,
        obstacle_distance_m: Optional[float] = None,
    ) -> VelocityCommand:
        ...


class UdpControlService:
    """Send commands at a fixed rate and send zero when stopping."""

    def __init__(
        self,
        control: ControlLoop,
        sender: UdpCommandSender,
        frame_reader: FrameReader,
        intent_provider: Optional[Callable[[float], Optional[State]]] = None,
        obstacle_provider: Optional[Callable[[float], Optional[float]]] = None,
        tick_period_s: float = 0.05,
        frame_timeout_s: float = 2.0,
    ):
        if (
            isinstance(tick_period_s, bool)
            or not isinstance(tick_period_s, Real)
            or not math.isfinite(tick_period_s)
            or tick_period_s <= 0.0
        ):
            raise ValueError("tick period must be positive")
        if (
            isinstance(frame_timeout_s, bool)
            or not isinstance(frame_timeout_s, Real)
            or not math.isfinite(frame_timeout_s)
            or frame_timeout_s <= 0.0
        ):
            raise ValueError("frame timeout must be positive")
        self.control = control
        self.sender = sender
        self.frame_reader = frame_reader
        self.intent_provider = intent_provider or (lambda timestamp_s: None)
        self.obstacle_provider = obstacle_provider or (lambda timestamp_s: None)
        self.tick_period_s = tick_period_s
        self.frame_timeout_s = frame_timeout_s

    async def run(self, stop_event: asyncio.Event):
        """Run until stopped or until the video is lost."""

        self.sender.start()
        last_frame_at_s = None
        try:
            while not stop_event.is_set():
                sample = await self.frame_reader()
                if sample is None:
                    raise RuntimeError("video stream ended before control shutdown")
                timestamp_s, frame = sample
                if last_frame_at_s is None:
                    last_frame_at_s = timestamp_s
                elif frame is not None:
                    last_frame_at_s = timestamp_s
                elif timestamp_s - last_frame_at_s > self.frame_timeout_s:
                    raise RuntimeError("video stream stalled before control shutdown")
                command = self.control.tick(
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
