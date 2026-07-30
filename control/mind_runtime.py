"""Run the Mac brain beside the fixed-rate motion loop."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

from control.mind import ConsciousDecision, MacMind, Telemetry
from control.mind_motion import movement_command
from control.watchdog import SetpointWatchdog
from control.velocity import VelocityCommand


class MindRuntime:
    """Turn VLM suggestions into safe Mac-side commands."""

    def __init__(
        self,
        mind: MacMind,
    ):
        self.mind = mind
        self.watchdog = SetpointWatchdog()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        self._observation = None
        self._decision: Optional[ConsciousDecision] = None
        self._closed = False

    @property
    def latest_decision(self) -> Optional[ConsciousDecision]:
        """Return the newest conscious decision, if one exists."""

        return self._decision

    def tick(
        self,
        frame,
        timestamp_s: float,
        intent: Optional[str] = None,
        obstacle_distance_m: Optional[float] = None,
    ) -> VelocityCommand:
        """Run one subconscious step and return one safe command."""

        if intent is not None:
            self.mind.set_intent(intent)
        self._collect()
        if frame is not None and self._future is None and not self._closed:
            self._future = self._executor.submit(
                self.mind.see,
                frame,
                timestamp_s,
                Telemetry(obstacle_distance_m=obstacle_distance_m),
            )
        movement = "stop"
        if self._observation is not None:
            age_s = timestamp_s - self._observation.timestamp_s
            if 0.0 <= age_s <= 0.5:
                movement = self._observation.movement
        desired = movement_command(movement, obstacle_distance_m)
        return self.watchdog.emit(timestamp_s, desired)

    def _collect(self):
        if self._future is None or not self._future.done():
            return
        future = self._future
        self._future = None
        self._observation = future.result()

    async def think_loop(
        self,
        stop_event: asyncio.Event,
        telemetry_provider: Callable[[], Telemetry],
        dialogue_provider: Optional[Callable[[], Optional[str]]] = None,
        period_s: float = 1.0,
    ):
        """Run conscious thoughts independently of image processing."""

        if period_s <= 0.0:
            raise ValueError("thinking period must be positive")
        while not stop_event.is_set():
            dialogue = dialogue_provider() if dialogue_provider is not None else None
            self._decision = await asyncio.to_thread(
                self.mind.think,
                telemetry_provider(),
                dialogue,
            )
            await asyncio.sleep(period_s)

    def close(self):
        """Stop background VLM work."""

        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
