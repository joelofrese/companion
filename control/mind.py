"""The Mac's subconscious and conscious brain."""

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class Telemetry:
    """The vehicle information the brain can use."""

    obstacle_distance_m: Optional[float] = None
    battery_percent: Optional[float] = None


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
    intent: str
    previous_movement: str
    dialogue: Optional[str]
    telemetry: Telemetry


@dataclass(frozen=True)
class ConsciousDecision:
    """The result of one conscious thought."""

    intent: str
    focus: str = ""
    dialogue: str = ""
    summary: str = ""


@dataclass
class MindState:
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

    def __init__(self, visual_model: VisualModel, language_model: LanguageModel):
        self.visual_model = visual_model
        self.language_model = language_model
        self.state = MindState()
        self._new_observations = deque[VisualObservation](maxlen=8)
        self._lock = threading.Lock()

    def set_intent(self, intent: str):
        """Set the high-level intent seen by the subconscious."""

        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("intent must be a non-empty string")
        with self._lock:
            self.state.intent = intent

    def see(
        self,
        image: Any,
        timestamp_s: float,
        telemetry: Telemetry = Telemetry(),
    ) -> VisualObservation:
        """Ask the VLM to describe one image."""

        with self._lock:
            focus = self.state.focus
            intent = self.state.intent
            previous_movement = self.state.previous_movement
            previous_observation = self.state.previous_observation
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
            self.state.previous_movement = observation.movement
            self.state.previous_observation = observation.description
            self._new_observations.append(observation)
        return observation

    def think(
        self,
        telemetry: Telemetry,
        dialogue: Optional[str] = None,
    ) -> ConsciousDecision:
        """Ask the LLM to update intent, focus, dialogue, and summary."""

        with self._lock:
            information = ConsciousInput(
                new_observations=tuple(self._new_observations),
                summary=self.state.summary,
                intent=self.state.intent,
                previous_movement=self.state.previous_movement,
                dialogue=dialogue,
                telemetry=telemetry,
            )
            self._new_observations.clear()
        decision = self.language_model.think(information)
        if not isinstance(decision.intent, str) or not decision.intent.strip():
            raise ValueError("language model returned an empty intent")
        with self._lock:
            self.state.intent = decision.intent
            self.state.focus = decision.focus
            self.state.summary = decision.summary
        return decision
