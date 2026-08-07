"""Run the Mac brain beside the fixed-rate motion loop."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import time
from typing import Callable, Optional

from control.mind import ConsciousDecision, MacMind, Telemetry, VisualObservation
from control.mind_motion import movement_command
from control.watchdog import SetpointWatchdog
from control.velocity import VelocityCommand
from voice.intent import parse_intent


MIN_MOVEMENT_CONFIDENCE = 0.5
MAX_MOVEMENT_AGE_S = 1.5
MAX_FRAME_GAP_S = 0.5
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
        self._observation_ready_at_s: Optional[float] = None
        self._last_frame_at_s: Optional[float] = None
        self._observation_intent: Optional[str] = None
        self._decision: Optional[ConsciousDecision] = None
        self._thought_error: Optional[Exception] = None
        self._closed = False
        self._observation_count = 0
        self._decision_count = 0
        self._last_command: Optional[VelocityCommand] = None
        self._intent_override: Optional[str] = None

    @property
    def latest_decision(self) -> Optional[ConsciousDecision]:
        """Return the newest conscious decision, if one exists."""

        return self._decision

    @property
    def latest_observation(self) -> Optional[VisualObservation]:
        """Return the newest observation processed by the visual model."""

        return self._observation

    @property
    def latest_observation_age_s(self) -> Optional[float]:
        """Return how long the newest VLM result has been available."""

        if self._observation_ready_at_s is None:
            return None
        return max(0.0, time.monotonic() - self._observation_ready_at_s)

    @property
    def latest_frame_age_s(self) -> Optional[float]:
        """Return how long ago a fresh camera frame arrived."""

        if self._last_frame_at_s is None:
            return None
        return max(0.0, time.monotonic() - self._last_frame_at_s)

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
        telemetry: Telemetry = Telemetry(),
    ) -> VelocityCommand:
        """Run one subconscious step and return one safe command."""

        if self._closed:
            return VelocityCommand()
        if intent is not None:
            self.mind.set_intent(intent)
        self._collect()
        if frame is not None:
            self._last_frame_at_s = time.monotonic()
        if frame is not None and self._future is None and not self._closed:
            self._future_intent = self.mind.intent
            self._future = self._executor.submit(
                self.mind.see,
                frame,
                timestamp_s,
                replace(telemetry, last_command=self._last_command),
            )
        movement = "stop"
        if self._observation is not None:
            age_s = self.latest_observation_age_s
            frame_age_s = self.latest_frame_age_s
            if self._thought_error is None and (
                self._observation_intent == self.mind.intent
                and age_s is not None
                and age_s <= MAX_MOVEMENT_AGE_S
                and frame_age_s is not None
                and frame_age_s <= MAX_FRAME_GAP_S
                and self._observation.confidence >= MIN_MOVEMENT_CONFIDENCE
            ):
                movement = self._observation.movement
        # A visual suggestion is unsafe without fresh vehicle state.
        if any(
            value is None
            for value in (
                telemetry.north_velocity_m_s,
                telemetry.east_velocity_m_s,
                telemetry.down_velocity_m_s,
            )
        ):
            movement = "stop"
        if parse_intent(self.mind.intent) == "hover":
            movement = "stop"
        desired = movement_command(movement, telemetry.obstacle_distance_m)
        command = self.watchdog.emit(timestamp_s, desired)
        self._last_command = command
        return command

    def _collect(self):
        if self._future is None or not self._future.done():
            return
        future = self._future
        self._future = None
        try:
            observation = future.result()
        except Exception:
            # A failed frame cannot justify movement. Retry the next frame.
            self._observation = None
            self._observation_ready_at_s = None
            self._observation_intent = None
            self._future_intent = None
            return
        self._observation = observation
        self._observation_ready_at_s = time.monotonic()
        self._observation_count += 1
        self._observation_intent = self._future_intent
        self._future_intent = None

    async def _think_or_stop(
        self,
        stop_event: asyncio.Event,
        telemetry: Telemetry,
        dialogue: Optional[str],
        intent_override: Optional[str],
    ) -> Optional[ConsciousDecision]:
        thought = asyncio.create_task(
            asyncio.to_thread(
                self.mind.think,
                telemetry,
                dialogue,
                intent_override,
            )
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
            if dialogue:
                self._intent_override = explicit_intent
                if explicit_intent is not None:
                    self.mind.set_intent(explicit_intent)
            if (
                self._decision_count == 0
                and self._observation_intent != self.mind.intent
                and not dialogue
            ):
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=period_s)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                telemetry = replace(
                    telemetry_provider(),
                    last_command=self._last_command,
                )
                decision = await self._think_or_stop(
                    stop_event,
                    telemetry,
                    dialogue,
                    self._intent_override,
                )
            except Exception as error:
                if (
                    self._thought_error is None
                    or str(error) != str(self._thought_error)
                ):
                    print(
                        f"Conscious thought failed; holding zero and retrying: {error}",
                        flush=True,
                    )
                self._thought_error = error
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=period_s)
                except asyncio.TimeoutError:
                    pass
                continue
            if decision is None:
                return
            self._thought_error = None
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
