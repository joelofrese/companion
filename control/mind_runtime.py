"""Run the Mac brain beside the fixed-rate motion loop."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

from control.mind import ConsciousDecision, MacMind, Telemetry, VisualObservation
from control.mind_motion import movement_command
from control.watchdog import SetpointWatchdog
from control.velocity import VelocityCommand
from voice.intent import parse_intent


MIN_MOVEMENT_CONFIDENCE = 0.5
MAX_MOVEMENT_AGE_S = 1.5
CONSCIOUS_PERIOD_S = 0.5


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
        self._future_intent: Optional[str] = None
        self._observation = None
        self._observation_intent: Optional[str] = None
        self._decision: Optional[ConsciousDecision] = None
        self._error: Optional[Exception] = None
        self._closed = False
        self._observation_count = 0
        self._decision_count = 0

    @property
    def latest_decision(self) -> Optional[ConsciousDecision]:
        """Return the newest conscious decision, if one exists."""

        return self._decision

    @property
    def latest_observation(self) -> Optional[VisualObservation]:
        """Return the newest observation processed by the visual model."""

        return self._observation

    @property
    def observation_count(self) -> int:
        """Return the number of VLM observations completed."""

        return self._observation_count

    @property
    def decision_count(self) -> int:
        """Return the number of conscious thoughts completed."""

        return self._decision_count

    def tick(
        self,
        frame,
        timestamp_s: float,
        intent: Optional[str] = None,
        obstacle_distance_m: Optional[float] = None,
    ) -> VelocityCommand:
        """Run one subconscious step and return one safe command."""

        if self._error is not None:
            raise RuntimeError("conscious brain failed") from self._error
        if intent is not None:
            self.mind.set_intent(intent)
        self._collect()
        if frame is not None and self._future is None and not self._closed:
            self._future_intent = self.mind.intent
            self._future = self._executor.submit(
                self.mind.see,
                frame,
                timestamp_s,
                Telemetry(obstacle_distance_m=obstacle_distance_m),
            )
        movement = "stop"
        if frame is not None and self._observation is not None:
            age_s = timestamp_s - self._observation.timestamp_s
            if (
                self._observation_intent == self.mind.intent
                and 0.0 <= age_s <= MAX_MOVEMENT_AGE_S
                and self._observation.confidence >= MIN_MOVEMENT_CONFIDENCE
            ):
                movement = self._observation.movement
        if parse_intent(self.mind.intent) == "hover":
            movement = "stop"
        desired = movement_command(movement, obstacle_distance_m)
        return self.watchdog.emit(timestamp_s, desired)

    def _collect(self):
        if self._future is None or not self._future.done():
            return
        future = self._future
        self._future = None
        self._observation = future.result()
        self._observation_count += 1
        self._observation_intent = self._future_intent
        self._future_intent = None

    async def _think_or_stop(
        self,
        stop_event: asyncio.Event,
        telemetry: Telemetry,
        dialogue: Optional[str],
    ) -> Optional[ConsciousDecision]:
        thought = asyncio.create_task(
            asyncio.to_thread(self.mind.think, telemetry, dialogue)
        )
        stopped = asyncio.create_task(stop_event.wait())
        done, _ = await asyncio.wait(
            (thought, stopped),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stopped in done:
            await asyncio.gather(thought, return_exceptions=True)
            return None
        stopped.cancel()
        await asyncio.gather(stopped, return_exceptions=True)
        return thought.result()

    async def think_loop(
        self,
        stop_event: asyncio.Event,
        telemetry_provider: Callable[[], Telemetry],
        dialogue_provider: Optional[Callable[[], Optional[str]]] = None,
        period_s: float = CONSCIOUS_PERIOD_S,
    ):
        """Run conscious thoughts independently of image processing."""

        if period_s <= 0.0:
            raise ValueError("thinking period must be positive")
        while not stop_event.is_set():
            dialogue = dialogue_provider() if dialogue_provider is not None else None
            explicit_intent = parse_intent(dialogue) if dialogue else None
            if explicit_intent is not None:
                self.mind.set_intent(explicit_intent)
            try:
                decision = await self._think_or_stop(
                    stop_event,
                    telemetry_provider(),
                    dialogue,
                )
            except Exception as error:
                self._error = error
                return
            if decision is None:
                return
            self._decision = decision
            self._decision_count += 1
            if decision.dialogue:
                print(f"Companion: {decision.dialogue}", flush=True)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=period_s)
            except asyncio.TimeoutError:
                pass

    def close(self):
        """Finish background VLM work before closing."""

        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
