"""The Mac's subconscious and conscious brain."""

from collections import deque
from dataclasses import dataclass
import math
import threading
from typing import Any, Optional, Protocol

from control.memory import CompanionMemory
from control.velocity import VelocityCommand
from voice.intent import parse_focus, parse_intent


MAX_PENDING_OBSERVATIONS = 32
PLACEHOLDER_TEXT = frozenset(("stop", "camera", "camera frame"))
FOCUS_PLACEHOLDER_TEXT = PLACEHOLDER_TEXT | frozenset(
    ("forward", "backward", "left", "right", "up", "down", "hover")
)

@dataclass(frozen=True)
class Telemetry:
    """The vehicle information the brain can use."""

    obstacle_distance_m: Optional[float] = None
    last_command: Optional[VelocityCommand] = None
    forward_velocity_m_s: Optional[float] = None
    right_velocity_m_s: Optional[float] = None
    down_velocity_m_s: Optional[float] = None


def _experience_outcome(telemetry: Telemetry) -> str:
    """Return a short record of the last command and measured result."""

    command = telemetry.last_command
    if command is None:
        return ""

    def value(number):
        return (
            "?"
            if number is None or not math.isfinite(number)
            else f"{number:.2f}"
        )

    command_values = (
        command.forward_m_s,
        command.right_m_s,
        command.down_m_s,
    )
    velocity_values = (
        telemetry.forward_velocity_m_s,
        telemetry.right_velocity_m_s,
        telemetry.down_velocity_m_s,
    )
    return "; ".join(
        (
            f"obstacle={value(telemetry.obstacle_distance_m)}",
            f"command={','.join(value(number) for number in command_values)}",
            f"velocity={','.join(value(number) for number in velocity_values)}",
        )
    )


@dataclass(frozen=True)
class VisualObservation:
    """One VLM description of an image."""

    timestamp_s: float
    description: str
    focused_answer: str = ""
    movement: str = "stop"
    alternate_movement: str = ""
    next_focus: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class ConsciousInput:
    """The information given to one conscious thought."""

    new_observations: tuple[VisualObservation, ...]
    summary: str
    focus: str
    memory: str
    intent: str
    previous_movement: str
    dialogue: Optional[str]
    telemetry: Telemetry


@dataclass(frozen=True)
class ConsciousDecision:
    """The result of one conscious thought."""

    intent: str
    intent_changed: bool = False
    focus: str = ""
    dialogue: str = ""
    summary: str = ""


@dataclass
class MindMemory:
    """The small shared memory between the two brain layers."""

    focus: str = ""
    intent: str = "hover"
    summary: str = ""
    previous_movement: str = "stop"
    previous_observation: str = ""
    requested_focus: str = ""


class VisualModel(Protocol):
    def observe(
        self,
        image: Any,
        timestamp_s: float,
        focus: str,
        intent: str,
        previous_movement: str,
        previous_observation: str,
        telemetry: Telemetry,
    ) -> VisualObservation:
        ...


class LanguageModel(Protocol):
    def think(self, information: ConsciousInput) -> ConsciousDecision:
        ...


class MacMind:
    """Connect a VLM subconscious to a conscious language model."""

    def __init__(
        self,
        visual_model: VisualModel,
        language_model: LanguageModel,
        memory: Optional[CompanionMemory] = None,
    ):
        self.visual_model = visual_model
        self.language_model = language_model
        self.memory_store = memory
        self.memory = MindMemory()
        self._intent_generation = 0
        self._new_observations = deque[VisualObservation](
            maxlen=MAX_PENDING_OBSERVATIONS
        )
        self._lock = threading.Lock()
        self._closed = False

    def set_intent(self, intent: str):
        """Set the high-level intent seen by the subconscious."""

        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("intent must be a non-empty string")
        intent = intent.strip()
        with self._lock:
            focus = parse_focus(intent)
            if focus is None and parse_intent(intent) == "following":
                focus = "person"
            if _same_goal(intent, self.memory.intent) and (
                not focus or focus == self.memory.focus
            ):
                return
            self.memory.intent = intent
            self._invalidate_visual_context()
            if focus:
                self.memory.focus = focus

    def _invalidate_visual_context(self):
        """Discard visual context that belongs to the previous intent."""

        self._intent_generation += 1
        self.memory.focus = ""
        self.memory.summary = ""
        self.memory.previous_movement = "stop"
        self.memory.previous_observation = ""
        self.memory.requested_focus = ""
        self._new_observations.clear()

    @property
    def intent(self) -> str:
        """Return the current high-level intent."""

        with self._lock:
            return self.memory.intent

    @property
    def visual_focus(self) -> str:
        """Return the subject the next visual result must answer."""

        with self._lock:
            return self.memory.focus

    def see(
        self,
        image: Any,
        timestamp_s: float,
        telemetry: Telemetry = Telemetry(),
    ) -> VisualObservation:
        """Ask the VLM to describe one image."""

        with self._lock:
            intent_generation = self._intent_generation
            focus = self.memory.focus
            intent = self.memory.intent
            previous_movement = self.memory.previous_movement
            previous_observation = self.memory.previous_observation
        observation = self.visual_model.observe(
            image,
            timestamp_s=timestamp_s,
            focus=focus,
            intent=intent,
            previous_movement=previous_movement,
            previous_observation=previous_observation,
            telemetry=telemetry,
        )
        with self._lock:
            if not self._closed and self._intent_generation == intent_generation:
                self.memory.previous_movement = observation.movement
                self.memory.previous_observation = observation.description
                self._new_observations.append(observation)
        return observation

    def think(
        self,
        telemetry: Telemetry,
        dialogue: Optional[str] = None,
        intent_override: Optional[str] = None,
    ) -> ConsciousDecision:
        """Ask the LLM to update intent, focus, dialogue, and summary."""

        with self._lock:
            intent_generation = self._intent_generation
            information = ConsciousInput(
                new_observations=tuple(self._new_observations),
                summary=self.memory.summary,
                focus=self.memory.focus,
                memory=self.memory_store.context() if self.memory_store else "",
                intent=self.memory.intent,
                previous_movement=self.memory.previous_movement,
                dialogue=dialogue,
                telemetry=telemetry,
            )
            focus_answered = any(
                observation.focused_answer
                for observation in information.new_observations
            )
            pending_focus = self.memory.requested_focus
            self._new_observations.clear()
        try:
            decision = self.language_model.think(information)
            if intent_override is not None:
                if not isinstance(intent_override, str) or not intent_override.strip():
                    raise ValueError("intent override must be a non-empty string")
                candidate = intent_override.strip()
                intent = (
                    information.intent
                    if _same_goal(candidate, information.intent)
                    else candidate
                )
            elif decision.intent_changed is True:
                if not isinstance(decision.intent, str) or not decision.intent.strip():
                    raise ValueError("language model returned an empty intent")
                candidate = decision.intent.strip()
                intent = (
                    information.intent
                    if _same_goal(candidate, information.intent)
                    else candidate
                )
            else:
                # A continuing goal may be worded differently by the model.
                # Keep the current text so fresh VLM work remains usable.
                intent = information.intent
            intent_changed = intent != information.intent
            focus = decision.focus.strip() if isinstance(decision.focus, str) else ""
            requested_focus = parse_focus(information.dialogue or "")
            if requested_focus:
                focus = requested_focus
            elif intent_changed and not focus:
                focus = parse_focus(intent) or ""
            if focus.lower() in FOCUS_PLACEHOLDER_TEXT:
                focus = information.focus
            if not focus and information.new_observations:
                next_focus = information.new_observations[-1].next_focus
                if isinstance(next_focus, str):
                    next_focus = next_focus.strip()
                else:
                    next_focus = ""
                if next_focus.lower() not in FOCUS_PLACEHOLDER_TEXT:
                    focus = next_focus
            if not focus and not focus_answered:
                focus = information.focus
            summary = decision.summary.strip() if isinstance(decision.summary, str) else ""
            if not summary:
                summary = information.summary
            if not summary and information.new_observations:
                summary = information.new_observations[-1].description
            dialogue_response = (
                decision.dialogue.strip()
                if isinstance(decision.dialogue, str)
                else ""
            )
            if not dialogue_response and information.dialogue:
                dialogue_response = _acknowledgement(intent, requested_focus)
            answered_focus = next(
                (
                    observation.focused_answer
                    for observation in reversed(information.new_observations)
                    if observation.focused_answer
                ),
                "",
            )
            if not dialogue_response and pending_focus:
                dialogue_response = _focused_reply(answered_focus)
            decision = ConsciousDecision(
                intent=intent,
                intent_changed=intent_changed,
                focus=focus,
                dialogue=dialogue_response,
                summary=summary,
            )
        except Exception:
            with self._lock:
                if not self._closed and self._intent_generation == intent_generation:
                    pending = tuple(information.new_observations) + tuple(
                        self._new_observations
                    )
                    self._new_observations.clear()
                    self._new_observations.extend(
                        pending[-MAX_PENDING_OBSERVATIONS:]
                    )
            raise
        with self._lock:
            if self._closed or self._intent_generation != intent_generation:
                return ConsciousDecision(intent=self.memory.intent)
            if self.memory_store is not None and (
                information.new_observations or information.dialogue
            ):
                outcome = _experience_outcome(information.telemetry)
                entry = f"intent={intent}; focus={decision.focus or 'none'}"
                if outcome:
                    entry += f"; {outcome}"
                entry += f"; summary={summary}"
                if information.new_observations:
                    observation = information.new_observations[-1]
                    entry += f"; observed={observation.description}"
                    if observation.focused_answer:
                        entry += f"; answer={observation.focused_answer}"
                if isinstance(decision.dialogue, str) and decision.dialogue.strip():
                    entry += f"; response={decision.dialogue}"
                if information.dialogue:
                    entry += f"; user={information.dialogue}"
                self.memory_store.remember(entry)
            if (
                intent != self.memory.intent
                or decision.focus != self.memory.focus
            ):
                if intent != self.memory.intent:
                    self.memory.intent = intent
                self._invalidate_visual_context()
            self.memory.focus = decision.focus
            self.memory.summary = decision.summary
            if requested_focus:
                self.memory.requested_focus = requested_focus
            elif information.dialogue and parse_intent(information.dialogue):
                self.memory.requested_focus = ""
            elif (
                self.memory.requested_focus == pending_focus
                and answered_focus
            ):
                self.memory.requested_focus = ""
        return decision

    def close(self):
        """Discard model results that finish after shutdown."""

        with self._lock:
            self._closed = True
            self._new_observations.clear()


def _same_goal(first: str, second: str) -> bool:
    """Treat different words for one recognized goal as the same goal."""

    if first == second:
        return True
    first_kind = parse_intent(first)
    if first_kind is not None:
        return first_kind == parse_intent(second)
    first_focus = parse_focus(first)
    return bool(first_focus and first_focus == parse_focus(second))


def _acknowledgement(intent: str, focus: Optional[str]) -> str:
    """Return a short reply when the conscious model leaves dialogue blank."""

    intent_kind = parse_intent(intent)
    if intent_kind == "following":
        return "Following."
    if intent_kind == "hover":
        return "Hovering."
    if focus:
        return f"Looking for {focus}."
    if intent_kind == "exploring":
        return "Exploring."
    return ""


def _focused_reply(answer: str) -> str:
    """Return one focused visual result after a dialogue request."""

    return f"I see {answer}." if answer else ""
