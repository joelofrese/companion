"""Use local Ollama models for the Mac brain."""

import base64
from io import BytesIO
import json
import math
from numbers import Real
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from control.mind import (
    FOCUS_PLACEHOLDER_TEXT,
    PLACEHOLDER_TEXT,
    ConsciousDecision,
    ConsciousInput,
    Telemetry,
    VisualObservation,
)


MOVEMENTS = frozenset(("forward", "left", "right", "up", "down", "stop", "hover"))
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
MAX_OUTPUT_TOKENS = 64
MAX_IMAGE_SIDE = 640

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "focused_answer": {"type": "string"},
        "movement": {"type": "string", "enum": sorted(MOVEMENTS)},
        "next_focus": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "description",
        "focused_answer",
        "movement",
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


class OllamaClient:
    """Small standard-library client for one local Ollama server."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout_s: float = 60.0):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Ollama URL must not be empty")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, Real)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
        ):
            raise ValueError("Ollama timeout must be positive")
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_s = float(timeout_s)

    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = urljoin(self.base_url, path.lstrip("/"))
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed ({error.code}): {detail}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Ollama is unavailable at {self.base_url}") from error

    def check(self):
        """Fail early when the local server is not running."""

        self._request("/api/tags")

    def preload(self, model: str):
        """Load one model without generating a response."""

        if not isinstance(model, str) or not model.strip():
            raise ValueError("Ollama model must not be empty")
        self._request(
            "/api/generate",
            {
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": "5m",
                "options": {"num_predict": 1},
            },
        )

    def chat(
        self,
        model: str,
        prompt: str,
        schema: Dict[str, Any],
        image: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Ollama model must not be empty")
        message: Dict[str, Any] = {"role": "user", "content": prompt}
        if image is not None:
            message["images"] = [image]
        payload = {
            "model": model,
            "messages": [message],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": MAX_OUTPUT_TOKENS},
            "keep_alive": "5m",
        }
        if think is not None:
            payload["think"] = think
        response = self._request("/api/chat", payload)
        try:
            message = response["message"]
            contents = (message.get("content"), message.get("thinking"))
            for content in contents:
                if not isinstance(content, str) or not content.strip():
                    continue
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    continue
        except (AttributeError, KeyError, TypeError) as error:
            raise RuntimeError("Ollama returned invalid structured output") from error
        raise RuntimeError("Ollama returned invalid structured output")


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
You are the Companion Drone's subconscious visual system. Describe the current
camera frame and suggest one cautious next movement. Look broadly, but answer
the requested focus when one is present. Never suggest backward movement. Use
stop only when the scene or movement is uncertain, unsafe, or blocked. If the
requested thing is not visible but the scene is clear and safe, use a slow left
or right movement to look around it. The description must say what is visible,
not repeat a movement word. The focused answer must answer the requested focus,
not describe the movement. Never substitute another visible object for the
requested focus; if the focus cannot be confirmed, say it is not visible.
Return only the requested JSON.

Current high-level intent: {intent or "none"}
Requested visual focus: {focus or "none"}
Previous movement: {previous_movement or "stop"}
Previous description: {previous_observation or "none"}
Last requested command: {telemetry.last_command or "none yet"}
Measured body velocity:
forward={telemetry.forward_velocity_m_s}
right={telemetry.right_velocity_m_s}
down={telemetry.down_velocity_m_s}
Forward TOF distance: {telemetry.obstacle_distance_m}

The movement must be one of: forward, left, right, up, down, stop, hover.
Use measured velocity to distinguish the requested movement from what the
vehicle actually did. The description should be short plain English.
Confidence must be between 0 and 1. The next focus should be a short thing
worth checking next, or empty.
""".strip()
        data = self.client.chat(
            self.model,
            prompt,
            VISION_SCHEMA,
            image=_image_base64(image),
            think=False,
        )
        movement = _text(data, "movement").lower()
        if movement not in MOVEMENTS:
            movement = "stop"
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
            or description_key == intent_key
            or "unclear" in description_key
        ):
            description = "the scene is unclear"
            focused_answer = ""
            movement = "stop"
            confidence = 0.0
        return VisualObservation(
            timestamp_s=timestamp_s,
            description=description,
            focused_answer=focused_answer,
            movement=movement,
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
        return ConsciousDecision(
            intent=intent,
            intent_changed=data.get("intent_changed") is True,
            focus=_text(data, "focus"),
            dialogue=_text(data, "dialogue"),
            summary=_text(data, "summary"),
        )
