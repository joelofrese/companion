"""Run the deterministic simulation mind beside the motion loop."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import time
from typing import Callable, Optional

from control.mind import ConsciousDecision, CompanionMind, Telemetry, VisualObservation
from control.mind_motion import movement_command
from control.velocity import VelocityCommand
from voice.intent import parse_intent


MIN_MOVEMENT_CONFIDENCE = 0.5
MAX_MOVEMENT_AGE_S = 1.5
MAX_FRAME_GAP_S = 0.5
CONSCIOUS_PERIOD_S = 0.5


class MindRuntime:
    """Turn VLM suggestions into safe companion commands."""

    def __init__(
        self,
        mind: CompanionMind,
    ):
        self.mind = mind
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        self._future_intent: Optional[str] = None
        self._future_focus: Optional[str] = None
        self._future_started_at_s: Optional[float] = None
        self._observation = None
        self._observation_ready_at_s: Optional[float] = None
        self._observation_duration_s: Optional[float] = None
        self._last_frame_at_s: Optional[float] = None
        self._observation_intent: Optional[str] = None
        self._observation_focus: Optional[str] = None
        self._decision: Optional[ConsciousDecision] = None
        self._decision_duration_s: Optional[float] = None
        self._thought_error: Optional[Exception] = None
        self._closed = False
        self._observation_count = 0
        self._decision_count = 0
        self._last_command: Optional[VelocityCommand] = None
        self._dialogue_intent: Optional[str] = None
        self._pending_dialogue: Optional[str] = None

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

    @property
    def latest_observation_duration_s(self) -> Optional[float]:
        """Return how long the newest VLM request took."""

        return self._observation_duration_s

    @property
    def latest_decision_duration_s(self) -> Optional[float]:
        """Return how long the newest conscious thought took."""

        return self._decision_duration_s

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
            self._future_focus = self.mind.visual_focus
            self._future_started_at_s = time.monotonic()
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
            observation_fresh = self._thought_error is None and (
                self._observation_intent == self.mind.intent
                and self._observation_focus == self.mind.visual_focus
                and age_s is not None
                and age_s <= MAX_MOVEMENT_AGE_S
                and frame_age_s is not None
                and frame_age_s <= MAX_FRAME_GAP_S
                and self._observation.confidence >= MIN_MOVEMENT_CONFIDENCE
            )
            if observation_fresh:
                movement = self._observation.movement
                alternate = self._observation.alternate_movement
                if (
                    alternate in ("left", "right")
                    and movement in ("forward", "stop")
                    and parse_intent(self.mind.intent) in ("following", "exploring")
                ):
                    movement = alternate
            if (
                self._observation.focused_answer
                and (
                    self.mind.awaiting_focus_answer
                    or parse_intent(self.mind.intent) is None
                )
            ):
                movement = "stop"
        # A visual suggestion is unsafe without fresh vehicle state.
        if any(
            value is None
            for value in (
                telemetry.forward_velocity_m_s,
                telemetry.right_velocity_m_s,
                telemetry.down_velocity_m_s,
            )
        ):
            movement = "stop"
        if parse_intent(self.mind.intent) == "hover":
            movement = "stop"
        desired = movement_command(movement, telemetry.obstacle_distance_m)
        self._last_command = desired
        return desired

    def _collect(self):
        if self._future is None or not self._future.done():
            return
        future = self._future
        self._future = None
        started_at_s = self._future_started_at_s
        self._future_started_at_s = None
        if started_at_s is not None:
            self._observation_duration_s = max(0.0, time.monotonic() - started_at_s)
        try:
            observation = future.result()
        except Exception:
            # A failed frame cannot justify movement. Retry the next frame.
            self._observation = None
            self._observation_ready_at_s = None
            self._observation_intent = None
            self._observation_focus = None
            self._future_intent = None
            self._future_focus = None
            return
        self._observation = observation
        self._observation_ready_at_s = time.monotonic()
        self._observation_count += 1
        self._observation_intent = self._future_intent
        self._observation_focus = self._future_focus
        self._future_intent = None
        self._future_focus = None

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
            thought.cancel()
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
            new_dialogue = (
                dialogue_provider() if dialogue_provider is not None else None
            )
            if new_dialogue:
                self._pending_dialogue = new_dialogue
                self._dialogue_intent = parse_intent(new_dialogue)
            dialogue = self._pending_dialogue
            if dialogue and self._dialogue_intent is not None:
                self.mind.set_intent(dialogue)
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
                started_at_s = time.monotonic()
                telemetry = replace(
                    telemetry_provider(),
                    last_command=self._last_command,
                )
                decision = await self._think_or_stop(
                    stop_event,
                    telemetry,
                    dialogue,
                    self._dialogue_intent,
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
            self._pending_dialogue = None
            self._thought_error = None
            self._decision_duration_s = max(0.0, time.monotonic() - started_at_s)
            self._decision = decision
            self._decision_count += 1
            if decision.dialogue:
                print(f"Companion: {decision.dialogue}", flush=True)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=period_s)
            except asyncio.TimeoutError:
                pass

    def close(self):
        """Stop new work so later ticks return zero immediately."""

        if self._closed:
            return
        self._closed = True
        self.mind.close()
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
