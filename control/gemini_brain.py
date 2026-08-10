"""Run Gemini Robotics ER as one streaming Mac brain."""

import asyncio
from collections import deque
from io import BytesIO
import math
import os
import time
from typing import Optional

from PIL import Image

from control.memory import CompanionMemory
from control.mind import ConsciousDecision, Telemetry, VisualObservation
from control.mind_motion import movement_command
from control.velocity import VelocityCommand
from voice.intent import parse_intent


DEFAULT_MODEL = "gemini-robotics-er-2-streaming-preview"
MIN_HEARTBEAT_PERIOD_S = 1.0
RESPONSE_TIMEOUT_S = 10.0
START_TIMEOUT_S = 20.0
MIN_MOVE_S = 0.2
MAX_MOVE_S = 1.0
MAX_IMAGE_WIDTH = 640


class GeminiRuntime:
    """Give one streaming Gemini session a deliberately small body."""

    def __init__(
        self,
        intent: str,
        model: str = DEFAULT_MODEL,
        memory: Optional[CompanionMemory] = None,
        api_key: Optional[str] = None,
    ):
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("intent must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Gemini model must be a non-empty string")
        self.intent = intent.strip()
        self.model = model.strip()
        self.memory_store = memory
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._latest_frame = None
        self._telemetry = Telemetry()
        self._dialogue = deque()
        self._movement = "stop"
        self._movement_until_s = 0.0
        self._movement_blocked = False
        self.latest_observation: Optional[VisualObservation] = None
        self.latest_decision: Optional[ConsciousDecision] = None
        self._last_heartbeat_at_s: Optional[float] = None
        self.latest_observation_duration_s: Optional[float] = None
        self.latest_decision_duration_s: Optional[float] = None
        self.observation_count = 0
        self.decision_count = 0
        self._response_parts = []
        self._closed = asyncio.Event()
        self._frame_ready = asyncio.Event()
        self._ready = asyncio.Event()
        self._task = None
        self._error: Optional[Exception] = None

    async def start(self):
        """Connect the persistent Gemini session before flight begins."""

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini")
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=START_TIMEOUT_S)
        except asyncio.TimeoutError as error:
            self.close()
            raise RuntimeError(
                "Gemini did not connect before the start timeout"
            ) from error
        if self._error is not None:
            raise RuntimeError(
                f"Gemini did not connect: {self._error}"
            ) from self._error

    def set_intent(self, intent: str):
        """Set the current goal included in the next heartbeat."""

        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("intent must be a non-empty string")
        self.intent = intent.strip()

    def add_dialogue(self, message: str):
        """Deliver one user request in the next model heartbeat."""

        if not isinstance(message, str) or not message.strip():
            return
        message = message.strip()
        self._dialogue.append(message)
        if parse_intent(message) == "hover":
            self.set_intent(message)
            self._movement = "stop"
            self._movement_until_s = 0.0

    def tick(
        self,
        frame,
        timestamp_s: float,
        intent: Optional[str] = None,
        telemetry: Telemetry = Telemetry(),
    ) -> VelocityCommand:
        """Store fresh state and return the current bounded action."""

        if self._closed.is_set() or self._error is not None:
            return VelocityCommand()
        if intent is not None:
            self.set_intent(intent)
        if frame is not None:
            self._latest_frame = frame
            self._frame_ready.set()
        self._telemetry = telemetry
        if time.monotonic() >= self._movement_until_s:
            self._movement = "stop"
        if any(
            value is None or not math.isfinite(value)
            for value in (
                telemetry.forward_velocity_m_s,
                telemetry.right_velocity_m_s,
                telemetry.down_velocity_m_s,
            )
        ):
            command = VelocityCommand()
        else:
            command = movement_command(self._movement, telemetry.obstacle_distance_m)
        if self._movement != "stop" and command == VelocityCommand():
            self._movement_blocked = True
        return command

    def close(self):
        """Stop movement and end the streaming session."""

        self._movement = "stop"
        self._movement_until_s = 0.0
        self._closed.set()
        self._frame_ready.set()

    async def wait_closed(self):
        """Wait for the session task after closing it."""

        if self._task is not None:
            await self._task

    async def _run(self):
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            config = types.LiveConnectConfig(
                response_modalities=["TEXT"],
                tools=_tools(),
                system_instruction=_system_instruction(),
            )
            async with client.aio.live.connect(
                model=self.model,
                config=config,
            ) as session:
                self._ready.set()
                await self._frame_ready.wait()
                if self._closed.is_set():
                    return
                while not self._closed.is_set():
                    sent_at_s = time.monotonic()
                    await self._heartbeat(session, types)
                    try:
                        await asyncio.wait_for(
                            self._receive(session, types), timeout=RESPONSE_TIMEOUT_S
                        )
                    except asyncio.TimeoutError:
                        pass
                    remaining_s = MIN_HEARTBEAT_PERIOD_S - (
                        time.monotonic() - sent_at_s
                    )
                    if remaining_s > 0.0:
                        try:
                            await asyncio.wait_for(
                                self._closed.wait(), timeout=remaining_s
                            )
                        except asyncio.TimeoutError:
                            pass
        except Exception as error:
            self._error = error
            self._ready.set()

    async def _heartbeat(self, session, types):
        frame = self._latest_frame
        if frame is not None:
            image_bytes = await asyncio.to_thread(_jpeg, frame)
            await session.send_realtime_input(
                video=types.Blob(data=image_bytes, mime_type="image/jpeg")
            )
        self._last_heartbeat_at_s = time.monotonic()
        dialogue = self._dialogue.popleft() if self._dialogue else ""
        await session.send_realtime_input(text=self._heartbeat_text(dialogue))

    def _heartbeat_text(self, dialogue: str) -> str:
        memory = self.memory_store.context() if self.memory_store is not None else ""
        state = (
            f"[HEARTBEAT] Goal: {self.intent}\n"
            f"Vehicle: {_telemetry_text(self._telemetry)}\n"
            "Inspect the latest image. If uncertain, hover. Use a tool for every "
            "physical action. Never infer that a move is safe from the image alone. "
            "Reply to every heartbeat with one concise observation and decision."
        )
        if dialogue:
            state += f"\nUser: {dialogue}"
        if memory:
            state += f"\nMemory:\n{memory}"
        return state

    async def _receive(self, session, types):
        async for message in session.receive():
            content = message.server_content
            if content is not None:
                transcript = content.output_transcription
                if transcript is not None and transcript.text:
                    self._response_parts.append(transcript.text)
                else:
                    turn = content.model_turn
                    if turn is not None and turn.parts:
                        self._response_parts.extend(
                            part.text for part in turn.parts if part.text
                        )
                if content.turn_complete:
                    text = "".join(self._response_parts).strip()
                    self._response_parts.clear()
                    if text:
                        self._record(text)
                        print(f"Companion: {text}", flush=True)
            tool_call = message.tool_call
            if tool_call is not None:
                responses = []
                for call in tool_call.function_calls:
                    result = await self._execute(call.name, call.args or {})
                    responses.append(
                        types.FunctionResponse(
                            name=call.name,
                            response=result,
                            id=call.id,
                        )
                    )
                await session.send_tool_response(function_responses=responses)

    async def _execute(self, name: str, args: dict) -> dict:
        if name == "move":
            return await self._move(args)
        if name == "hover":
            self._movement = "stop"
            self._movement_until_s = 0.0
            self._record("Chose to hover.")
            return {"status": "hovering", "telemetry": _telemetry_text(self._telemetry)}
        if name == "speak":
            message = str(args.get("message", "")).strip()
            if not message:
                return {"status": "rejected", "reason": "message is required"}
            self._record(message, dialogue=message)
            print(f"Companion: {message}", flush=True)
            return {"status": "spoken"}
        return {"status": "rejected", "reason": "unknown tool"}

    async def _move(self, args: dict) -> dict:
        direction = str(args.get("direction", "")).strip().lower()
        duration_s = args.get("duration_s")
        if direction not in ("forward", "left", "right"):
            return {
                "status": "rejected",
                "reason": "direction must be forward, left, or right",
            }
        if (
            isinstance(duration_s, bool)
            or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s)
            or not MIN_MOVE_S <= duration_s <= MAX_MOVE_S
        ):
            return {
                "status": "rejected",
                "reason": f"duration_s must be {MIN_MOVE_S} to {MAX_MOVE_S}",
            }
        self._movement = direction
        self._movement_until_s = time.monotonic() + duration_s
        self._movement_blocked = False
        self._record(f"Chose to move {direction} for {duration_s:.1f} seconds.")
        try:
            await asyncio.wait_for(self._closed.wait(), timeout=duration_s)
            return {"status": "cancelled"}
        except asyncio.TimeoutError:
            pass
        finally:
            self._movement = "stop"
            self._movement_until_s = 0.0
        status = "stopped by local safety" if self._movement_blocked else "completed"
        return {"status": status, "telemetry": _telemetry_text(self._telemetry)}

    def _record(self, text: str, dialogue: str = ""):
        text = " ".join(str(text).split())
        if not text:
            return
        now = time.monotonic()
        latency = (
            max(0.0, now - self._last_heartbeat_at_s)
            if self._last_heartbeat_at_s is not None
            else None
        )
        self.latest_observation = VisualObservation(
            timestamp_s=now,
            description=text,
            movement=self._movement,
            confidence=1.0,
        )
        self.latest_decision = ConsciousDecision(
            intent=self.intent,
            dialogue=dialogue,
            summary=text,
        )
        self.latest_observation_duration_s = latency
        self.latest_decision_duration_s = latency
        self.observation_count += 1
        self.decision_count += 1
        if self.memory_store is not None:
            self.memory_store.remember(
                f"intent={self.intent}; "
                f"{_telemetry_text(self._telemetry)}; summary={text}"
            )


def _tools():
    """Return the complete, bounded body exposed to Gemini."""

    return [{"function_declarations": [
        {
            "name": "move",
            "description": "Move slowly in one body direction for a short time.",
            "behavior": "BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "direction": {
                        "type": "STRING",
                        "enum": ["forward", "left", "right"],
                    },
                    "duration_s": {
                        "type": "NUMBER",
                        "description": "A duration from 0.2 through 1.0 seconds.",
                    },
                },
                "required": ["direction", "duration_s"],
            },
        },
        {
            "name": "hover",
            "description": "Stop horizontal motion and hold position.",
            "behavior": "BLOCKING",
            "parameters": {"type": "OBJECT", "properties": {}},
        },
        {
            "name": "speak",
            "description": "Say one short message to the nearby user.",
            "behavior": "BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {"message": {"type": "STRING"}},
                "required": ["message"],
            },
        },
    ]}]


def _system_instruction() -> str:
    """State the safety and control contract in plain language."""

    return (
        "You are the high-level brain of an indoor companion drone. Observe the "
        "latest camera image, user dialogue, goal, and telemetry. Think broadly, "
        "but move slowly and deliberately. You can only request brief forward or "
        "lateral translation; another computer checks every action and may stop it. "
        "Never request altitude, heading, motors, attitude, position, or a long "
        "move. Use hover whenever the scene or telemetry is unclear. Describe "
        "important observations briefly and speak when a user needs an answer."
    )


def _jpeg(frame) -> bytes:
    """Turn one Gazebo BGR frame into the bounded JPEG sent to Gemini."""

    image = Image.fromarray(frame[:, :, ::-1])
    if image.width > MAX_IMAGE_WIDTH:
        image.thumbnail((MAX_IMAGE_WIDTH, image.height))
    output = BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def _telemetry_text(telemetry: Telemetry) -> str:
    """Return the small telemetry record useful to the model and memory."""

    command = telemetry.last_command or VelocityCommand()
    return "; ".join(
        (
            f"obstacle={_number(telemetry.obstacle_distance_m)}",
            "command=" + ",".join(
                _number(value)
                for value in (
                    command.forward_m_s,
                    command.right_m_s,
                    command.down_m_s,
                )
            ),
            "velocity=" + ",".join(
                _number(value)
                for value in (
                    telemetry.forward_velocity_m_s,
                    telemetry.right_velocity_m_s,
                    telemetry.down_velocity_m_s,
                )
            ),
        )
    )


def _number(value) -> str:
    """Format one finite telemetry value without pretending unknown is zero."""

    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.2f}"
    return "?"
