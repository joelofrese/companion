"""The Mac's subconscious and conscious brain."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Optional, Protocol

from control.velocity import VelocityCommand


MAX_PENDING_OBSERVATIONS = 32
MAX_MEMORY_LINES = 64
MAX_MEMORY_CHARS = 240
MEMORY_CONTEXT_LINES = 8
PLACEHOLDER_TEXT = frozenset(("stop", "camera", "camera frame"))


class CompanionMemory:
    """Keep a small, editable record of past conscious decisions."""

    def __init__(self, path):
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ValueError("memory path must not be empty")
        self.path = Path(path).expanduser()
        self._lines = self._read()

    def _read(self):
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [
            line.strip()[:MAX_MEMORY_CHARS]
            for line in lines
            if line.strip()
        ][-MAX_MEMORY_LINES:]

    def context(self) -> str:
        """Return the newest memories for the conscious prompt."""

        return "\n".join(self._lines[-MEMORY_CONTEXT_LINES:])

    def remember(self, entry: str):
        """Save one new memory and keep the file bounded."""

        entry = " ".join(entry.split())[:MAX_MEMORY_CHARS]
        if not entry or (self._lines and self._lines[-1] == entry):
            return
        self._lines = (self._lines + [entry])[-MAX_MEMORY_LINES:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Telemetry:
    """The vehicle information the brain can use."""

    obstacle_distance_m: Optional[float] = None
    last_command: Optional[VelocityCommand] = None
    north_velocity_m_s: Optional[float] = None
    east_velocity_m_s: Optional[float] = None
    down_velocity_m_s: Optional[float] = None


def _experience_outcome(telemetry: Telemetry) -> str:
    """Return a short record of the last command and measured result."""

    command = telemetry.last_command
    if command is None:
        return ""

    def value(number):
        return "?" if number is None else f"{number:.2f}"

    command_values = (
        command.north_m_s,
        command.east_m_s,
        command.down_m_s,
    )
    velocity_values = (
        telemetry.north_velocity_m_s,
        telemetry.east_velocity_m_s,
        telemetry.down_velocity_m_s,
    )
    return (
        "; command="
        + ",".join(value(number) for number in command_values)
        + "; velocity="
        + ",".join(value(number) for number in velocity_values)
    )


@dataclass(frozen=True)
class VisualObservation:
    """One VLM description of an image."""

    timestamp_s: float
    description: str
    focused_answer: str = ""
    movement: str = "stop"
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

    def set_intent(self, intent: str):
        """Set the high-level intent seen by the subconscious."""

        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("intent must be a non-empty string")
        intent = intent.strip()
        with self._lock:
            if intent == self.memory.intent:
                return
            self.memory.intent = intent
            self._invalidate_visual_context()

    def _invalidate_visual_context(self):
        """Discard visual context that belongs to the previous intent."""

        self._intent_generation += 1
        self.memory.focus = ""
        self.memory.summary = ""
        self.memory.previous_movement = "stop"
        self.memory.previous_observation = ""
        self._new_observations.clear()

    @property
    def intent(self) -> str:
        """Return the current high-level intent."""

        with self._lock:
            return self.memory.intent

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
            if self._intent_generation == intent_generation:
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
            self._new_observations.clear()
        try:
            decision = self.language_model.think(information)
            intent_changed = decision.intent_changed is True
            if intent_override is not None:
                if not isinstance(intent_override, str) or not intent_override.strip():
                    raise ValueError("intent override must be a non-empty string")
                intent = intent_override.strip()
                intent_changed = intent != information.intent
            elif intent_changed:
                if not isinstance(decision.intent, str) or not decision.intent.strip():
                    raise ValueError("language model returned an empty intent")
                intent = decision.intent.strip()
            else:
                # A continuing goal may be worded differently by the model.
                # Keep the current text so fresh VLM work remains usable.
                intent = information.intent
            intent_changed = intent != information.intent
            focus = decision.focus.strip() if isinstance(decision.focus, str) else ""
            if focus.lower() in PLACEHOLDER_TEXT:
                focus = information.focus
            if not focus and information.new_observations:
                next_focus = information.new_observations[-1].next_focus
                if isinstance(next_focus, str):
                    next_focus = next_focus.strip()
                else:
                    next_focus = ""
                if next_focus.lower() not in PLACEHOLDER_TEXT:
                    focus = next_focus
            summary = decision.summary.strip() if isinstance(decision.summary, str) else ""
            if not summary:
                summary = information.summary
            decision = ConsciousDecision(
                intent=intent,
                intent_changed=intent_changed,
                focus=focus,
                dialogue=(
                    decision.dialogue.strip()
                    if isinstance(decision.dialogue, str)
                    else ""
                ),
                summary=summary,
            )
        except Exception:
            with self._lock:
                if self._intent_generation == intent_generation:
                    pending = tuple(information.new_observations) + tuple(
                        self._new_observations
                    )
                    self._new_observations.clear()
                    self._new_observations.extend(
                        pending[-MAX_PENDING_OBSERVATIONS:]
                    )
            raise
        with self._lock:
            if self._intent_generation != intent_generation:
                return ConsciousDecision(intent=self.memory.intent)
            if self.memory_store is not None and (
                information.new_observations or information.dialogue
            ):
                entry = (
                    f"intent={intent}; focus={decision.focus or 'none'}; "
                    f"summary={summary}"
                )
                entry += _experience_outcome(information.telemetry)
                if isinstance(decision.dialogue, str) and decision.dialogue.strip():
                    entry += f"; response={decision.dialogue}"
                if information.dialogue:
                    entry = f"user={information.dialogue}; {entry}"
                self.memory_store.remember(entry)
            if intent != self.memory.intent:
                self.memory.intent = intent
                self._invalidate_visual_context()
            self.memory.focus = decision.focus
            self.memory.summary = decision.summary
        return decision
