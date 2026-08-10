"""Use local Ollama models for the Mac brain."""

import base64
from io import BytesIO
import math
from numbers import Real
from typing import Any, Dict

from control.mind import (
    FOCUS_PLACEHOLDER_TEXT,
    PLACEHOLDER_TEXT,
    ConsciousDecision,
    ConsciousInput,
    Telemetry,
    VisualObservation,
)
from control.mind_motion import MOVEMENT_NAMES
from control.ollama_client import OllamaClient
from control.safety_limits import OBSTACLE_STOP_M


ALTERNATE_MOVEMENTS = frozenset(("left", "right"))
DESCRIPTION_PLACEHOLDERS = PLACEHOLDER_TEXT | frozenset(
    (
        "current high-level intent",
        "requested visual focus",
        "previous movement",
        "previous description",
        "last requested command",
        "measured body velocity",
        "forward tof distance",
        "the scene is clear",
        "the scene is clear and safe",
    )
)
PROMPT_ECHO_PREFIXES = (
    "intent:",
    "current intent:",
    "visual focus:",
    "requested visual focus:",
    "previous movement:",
    "previous description:",
    "previous visual summary:",
    "last command:",
    "last requested command:",
    "body velocity:",
    "measured body velocity:",
    "forward tof distance:",
)
MAX_IMAGE_SIDE = 640

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "focused_answer": {"type": "string"},
        "movement": {"type": "string", "enum": sorted(MOVEMENT_NAMES)},
        "alternate_movement": {
            "type": "string",
            "enum": sorted(ALTERNATE_MOVEMENTS | frozenset(("none",))),
        },
        "next_focus": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "description",
        "focused_answer",
        "movement",
        "alternate_movement",
        "next_focus",
        "confidence",
    ],
}

CONSCIOUS_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "intent_changed": {"type": "boolean"},
        "focus": {"type": "string"},
        "dialogue": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": [
        "intent",
        "intent_changed",
        "focus",
        "dialogue",
        "summary",
    ],
}


def _text(data: Dict[str, Any], name: str) -> str:
    value = data.get(name, "")
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return "" if value.lower() in {"", "{}", "[]", "null", "none", "empty", "n/a"} else value


def _meaningful_text(data: Dict[str, Any], name: str) -> str:
    value = _text(data, name)
    return "" if value.lower() in PLACEHOLDER_TEXT else value


def _prompt_echo(value: str) -> bool:
    """Return whether model text repeats a prompt field instead of the scene."""

    value = " ".join(value.lower().split())
    return any(value.startswith(prefix) for prefix in PROMPT_ECHO_PREFIXES)


def _confirmed_focus(answer: str, focus: str) -> str:
    """Keep an answer only when it confirms the requested focus."""

    focus_text = " ".join(focus.lower().split())
    answer_text = " ".join(answer.lower().split())
    if not focus_text or not answer_text:
        return "" if not focus_text else answer
    if any(
        phrase in answer_text
        for phrase in (
            "not visible",
            "cannot see",
            "can't see",
            "don't see",
            "do not see",
        )
    ) or f"no {focus_text}" in answer_text:
        return ""
    answer_words = {word.strip(".,!?;:") for word in answer_text.split()}
    if all(word in answer_words for word in focus_text.split()):
        return answer
    return ""


def _confidence(data: Dict[str, Any]) -> float:
    value = data.get("confidence", 0.0)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _image_base64(image: Any) -> str:
    """Encode one BGR frame for Ollama's image field."""

    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(image).decode("ascii")

    from PIL import Image

    if hasattr(image, "save"):
        picture = image
    else:
        shape = getattr(image, "shape", ())
        if len(shape) == 3 and shape[-1] == 3:
            image = image[:, :, ::-1]
        picture = Image.fromarray(image)
    if max(picture.size) > MAX_IMAGE_SIDE:
        picture = picture.copy()
        picture.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = BytesIO()
    picture.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OllamaVisionModel:
    """Use one local vision model for the subconscious."""

    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

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
        prompt = f"""
You are the Companion Drone's visual mind. Look carefully at the image and
return only the requested JSON. Describe the most obvious visible object,
person, wall, or open path in short plain English. If a visual focus is given,
answer that subject exactly; never substitute another object. If it is not
visible, say so. Suggest one cautious movement. Use stop when the image is
unclear, unsafe, or blocked. For exploration, choose slow forward, left, or
right movement when a clear path is visible. For following, move toward a
visible focused subject when the path is clear. If forward TOF is at or below
{OBSTACLE_STOP_M:.1f}m,
choose stop. Never move backward.
Return alternate_movement as left or right only when the forward path is
blocked and that side is visibly clear; otherwise return none.

Intent: {intent or "none"}
Visual focus: {focus or "none"}
Previous movement: {previous_movement or "stop"}
Previous description: {previous_observation or "none"}
Last command: {telemetry.last_command or "none yet"}
Body velocity: forward={telemetry.forward_velocity_m_s}, right={telemetry.right_velocity_m_s}, down={telemetry.down_velocity_m_s}
Forward TOF distance: {telemetry.obstacle_distance_m}

Movement must be one of: forward, left, right, stop, hover.
Confidence must be between 0 and 1. Choose a short next visual focus or leave
it empty.
""".strip()
        data = self.client.chat(
            self.model,
            prompt,
            VISION_SCHEMA,
            image=_image_base64(image),
            think=False,
        )
        movement = _text(data, "movement").lower()
        if movement not in MOVEMENT_NAMES:
            movement = "stop"
        alternate_movement = _text(data, "alternate_movement").lower()
        if alternate_movement not in ALTERNATE_MOVEMENTS:
            alternate_movement = ""
        description = _meaningful_text(data, "description")
        next_focus = _meaningful_text(data, "next_focus")
        if next_focus.lower() in FOCUS_PLACEHOLDER_TEXT:
            next_focus = ""
        focused_answer = _confirmed_focus(
            _meaningful_text(data, "focused_answer"),
            focus,
        )
        confidence = _confidence(data)
        description_key = " ".join(description.lower().split()).strip(" .!?")
        intent_key = " ".join(intent.lower().split()).strip(" .!?")
        if (
            not description
            or description_key in DESCRIPTION_PLACEHOLDERS
            or any(
                description_key.startswith(f"{placeholder}:")
                for placeholder in DESCRIPTION_PLACEHOLDERS
            )
            or _prompt_echo(description)
            or description_key == intent_key
            or "unclear" in description_key
        ):
            description = "the scene is unclear"
            focused_answer = ""
            movement = "stop"
            alternate_movement = ""
            confidence = 0.0
        return VisualObservation(
            timestamp_s=timestamp_s,
            description=description,
            focused_answer=focused_answer,
            movement=movement,
            alternate_movement=alternate_movement,
            next_focus=next_focus,
            confidence=confidence,
        )


class OllamaLanguageModel:
    """Use one local language model for the conscious mind."""

    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def think(self, information: ConsciousInput) -> ConsciousDecision:
        observation_lines = []
        for observation in information.new_observations:
            line = (
                f"- {observation.description}; "
                f"focused answer={observation.focused_answer or 'none'}; "
                f"suggested movement={observation.movement}; "
                f"alternate={observation.alternate_movement or 'none'}; "
                f"focus next={observation.next_focus or 'none'}; "
                f"confidence={observation.confidence}"
            )
            if not observation_lines or line != observation_lines[-1]:
                observation_lines.append(line)
        observations = "\n".join(observation_lines) or (
            "(no new visual observations)"
        )
        prompt = f"""
You are the Companion Drone's conscious mind. Use the visual observations,
memory, telemetry, and optional dialogue to choose the next high-level intent.
Continue the current goal when it still makes sense, but think proactively:
choose what to notice or do next when the world gives you a reason.
Keep a clear current visual focus until it is answered, not visible, or no
longer relevant; do not replace it just because another object is visible.
When dialogue asks you to find or inspect something, make that subject the
focus even if the first frame is unclear.
Set intent_changed true only when the high-level goal really changes. When
continuing the current goal, set it false and copy the current intent exactly;
do not rephrase it. You may choose a short natural intent, but do not issue
motor or velocity commands. Keep intent under six words, focus under four
words, summary under twelve words, and dialogue under twelve words. Leave
dialogue empty unless a user deserves a response. Do not ask questions. Return
only the requested JSON.

Current intent: {information.intent}
Current visual focus: {information.focus or "none"}
Previous movement: {information.previous_movement}
Previous visual summary: {information.summary or "none"}
Long-term experience memory:
{information.memory or "none"}
New visual observations:
{observations}
Optional user dialogue: {information.dialogue or "none"}
Last requested command: {information.telemetry.last_command or "none yet"}
Measured body velocity:
forward={information.telemetry.forward_velocity_m_s}
right={information.telemetry.right_velocity_m_s}
down={information.telemetry.down_velocity_m_s}
Forward TOF distance: {information.telemetry.obstacle_distance_m}

Treat measured velocity as what actually happened. Compare it with the last
requested command and adapt the next intent when the vehicle did not respond
as expected.
The summary should stay short and describe what the drone currently knows.
""".strip()
        data = self.client.chat(
            self.model,
            prompt,
            CONSCIOUS_SCHEMA,
            think=False,
        )
        intent = _text(data, "intent") or information.intent
        focus = _meaningful_text(data, "focus")
        summary = _meaningful_text(data, "summary")
        if _prompt_echo(focus):
            focus = ""
        if _prompt_echo(summary):
            summary = ""
        return ConsciousDecision(
            intent=intent,
            intent_changed=data.get("intent_changed") is True,
            focus=focus,
            dialogue=_text(data, "dialogue"),
            summary=summary,
        )
