"""Run Gemini Robotics ER as the companion's streaming brain."""

import asyncio
from collections import deque
from dataclasses import dataclass
from io import BytesIO
import math
import os
import time
from typing import Optional

from PIL import Image

from control.memory import CompanionMemory
from control.mind import ConsciousDecision, Telemetry, VisualObservation
from control.mind_motion import movement_command
from control.safety_limits import MAX_YAW_RATE_DEG_S, OBSTACLE_STOP_M
from control.velocity import VelocityCommand


DEFAULT_MODEL = "gemini-robotics-er-2-streaming-preview"
DEFAULT_SITUATION = "Observe the indoor environment and decide what to do next."
VIDEO_PERIOD_S = 1.0
MIN_HEARTBEAT_PERIOD_S = 1.0
RESPONSE_TIMEOUT_S = 10.0
START_TIMEOUT_S = 20.0
RECONNECT_DELAY_S = 1.0
MIN_MOVE_S = 0.2
MAX_MOVE_S = 1.0
MIN_TURN_DEG = 15.0
MAX_TURN_DEG = 90.0
# Keep the yaw rate low enough for PX4 to settle near the requested heading.
TURN_RATE_DEG_S = 20.0
MAX_IMAGE_WIDTH = 640
ACTION_GRACE_S = 1.0
ACTION_SETTLE_S = 1.0
ACTION_STABLE_S = 0.3
HEADING_STABILITY_RAD = math.radians(2.0)
HEADING_TOLERANCE_RAD = math.radians(5.0)


@dataclass
class ActiveAction:
    """One physical action that must finish before another can start."""

    kind: str
    direction: str
    amount: float
    deadline_s: float
    start_heading_rad: Optional[float] = None
    phase: str = "running"
    stable_since_s: Optional[float] = None
    last_heading_rad: Optional[float] = None


class GeminiRuntime:
    """Give one streaming Gemini session a deliberately small body."""

    def __init__(
        self,
        situation: str = DEFAULT_SITUATION,
        model: str = DEFAULT_MODEL,
        memory: Optional[CompanionMemory] = None,
        api_key: Optional[str] = None,
    ):
        if not isinstance(situation, str) or not situation.strip():
            raise ValueError("situation must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Gemini model must be a non-empty string")
        self.situation = situation.strip()
        self.model = model.strip()
        self.memory_store = memory
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._latest_frame = None
        self._telemetry = Telemetry()
        self._dialogue = deque()
        self._movement = "stop"
        self._turn_direction = "stop"
        self._active_action: Optional[ActiveAction] = None
        self._last_action_result = ""
        self.latest_observation: Optional[VisualObservation] = None
        self.latest_decision: Optional[ConsciousDecision] = None
        self.latest_thought = ""
        self.latest_response = ""
        self.latest_action = "stop"
        self._last_heartbeat_at_s: Optional[float] = None
        self.latest_observation_duration_s: Optional[float] = None
        self.latest_decision_duration_s: Optional[float] = None
        self.observation_count = 0
        self.decision_count = 0
        self.turn_count = 0
        self.video_frame_count = 0
        self._response_parts = []
        self._response_thoughts = []
        self._action_summary = ""
        self.thought_count = 0
        self.thought_token_count = 0
        self._memory_sent = False
        self._bootstrap_pending = True
        self._session_handle: Optional[str] = None
        self.session_reconnect_count = 0
        self._reconnect_requested = False
        self._closed = asyncio.Event()
        self._frame_ready = asyncio.Event()
        self._send_lock = asyncio.Lock()
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

    def add_dialogue(self, message: str):
        """Deliver one user request in the next model heartbeat."""

        if not isinstance(message, str) or not message.strip():
            return
        self._dialogue.append(message.strip())

    def tick(
        self,
        frame,
        timestamp_s: float,
        telemetry: Telemetry = Telemetry(),
    ) -> VelocityCommand:
        """Store fresh state and return the current bounded action."""

        if self._closed.is_set() or self._error is not None:
            return VelocityCommand()
        if frame is not None:
            self._latest_frame = frame
            self._frame_ready.set()
        self._telemetry = telemetry
        self._refresh_action()
        if any(
            value is None or not math.isfinite(value)
            for value in (
                telemetry.forward_velocity_m_s,
                telemetry.right_velocity_m_s,
                telemetry.down_velocity_m_s,
            )
        ):
            command = VelocityCommand()
        elif self._active_action is not None:
            movement = movement_command(self._movement, telemetry.obstacle_distance_m)
            yaw_rate = 0.0
            if (
                self._turn_direction != "stop"
                and telemetry.obstacle_distance_m is not None
                and math.isfinite(telemetry.obstacle_distance_m)
                and telemetry.obstacle_distance_m > OBSTACLE_STOP_M
            ):
                yaw_rate = TURN_RATE_DEG_S * (
                    1.0 if self._turn_direction == "right" else -1.0
                )
            command = VelocityCommand(
                movement.forward_m_s,
                movement.right_m_s,
                movement.down_m_s,
                yaw_rate,
            )
        else:
            command = VelocityCommand()
        return command

    def close(self):
        """Stop movement and end the streaming session."""

        self._cancel_action("brain closed")
        self._movement = "stop"
        self._turn_direction = "stop"
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
            connected = False
            while not self._closed.is_set():
                self._reconnect_requested = False
                if self._session_handle is None:
                    self._memory_sent = False
                    self._bootstrap_pending = True
                try:
                    config = types.LiveConnectConfig(
                        response_modalities=["TEXT"],
                        tools=_tools(),
                        system_instruction=_system_instruction(),
                        thinking_config=types.ThinkingConfig(include_thoughts=True),
                        context_window_compression=(
                            types.ContextWindowCompressionConfig(
                                sliding_window=types.SlidingWindow()
                            )
                        ),
                        session_resumption=types.SessionResumptionConfig(
                            handle=self._session_handle
                        ),
                    )
                    async with client.aio.live.connect(
                        model=self.model,
                        config=config,
                    ) as session:
                        connected = True
                        self._ready.set()
                        await self._frame_ready.wait()
                        if self._closed.is_set():
                            return
                        await self._send_frame(session, types)
                        video_task = asyncio.create_task(
                            self._stream_video(session, types)
                        )
                        try:
                            while (
                                not self._closed.is_set()
                                and not self._reconnect_requested
                            ):
                                sent_at_s = time.monotonic()
                                await self._heartbeat(session)
                                try:
                                    await asyncio.wait_for(
                                        self._receive(session, types),
                                        timeout=RESPONSE_TIMEOUT_S,
                                    )
                                except asyncio.TimeoutError:
                                    pass
                                if video_task.done():
                                    await video_task
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
                        finally:
                            video_task.cancel()
                            await asyncio.gather(video_task, return_exceptions=True)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    if not connected:
                        raise error
                    if self._closed.is_set():
                        return
                    print(f"Gemini session reconnecting: {error}", flush=True)
                if self._closed.is_set():
                    return
                self.session_reconnect_count += 1
                try:
                    await asyncio.wait_for(
                        self._closed.wait(), timeout=RECONNECT_DELAY_S
                    )
                except asyncio.TimeoutError:
                    pass
        except Exception as error:
            self._error = error
            self._ready.set()

    async def _stream_video(self, session, types):
        """Keep the Live session supplied with the newest camera frame."""

        while not self._closed.is_set():
            try:
                await asyncio.wait_for(self._closed.wait(), timeout=VIDEO_PERIOD_S)
            except asyncio.TimeoutError:
                await self._send_frame(session, types)

    async def _send_frame(self, session, types):
        """Send one current JPEG without waiting for a model decision."""

        frame = self._latest_frame
        if frame is None:
            return
        image_bytes = await asyncio.to_thread(_jpeg, frame)
        async with self._send_lock:
            await session.send_realtime_input(
                video=types.Blob(data=image_bytes, mime_type="image/jpeg")
            )
        self.video_frame_count += 1

    async def _heartbeat(self, session):
        """Ask for one high-level decision while video keeps streaming."""

        self._last_heartbeat_at_s = time.monotonic()
        dialogue = self._dialogue.popleft() if self._dialogue else ""
        async with self._send_lock:
            await session.send_realtime_input(text=self._heartbeat_text(dialogue))

    def _heartbeat_text(self, dialogue: str) -> str:
        memory = ""
        if not self._memory_sent:
            self._memory_sent = True
            if self.memory_store is not None:
                memory = self.memory_store.context()
        start = ""
        if self._bootstrap_pending:
            self._bootstrap_pending = False
            start = f"[START]\nSituation: {self.situation}\n"
        state = (
            f"{start}[STATE]\n"
            f"Vehicle: {_telemetry_text(self._telemetry)}\n"
            f"Action: {self._action_state_text()}\n"
            "Inspect the latest image and current state. Decide for yourself whether "
            "to move, turn, hover, speak, or do nothing. Never infer that a move or "
            "turn is safe from the image alone. A heartbeat does not require a tool "
            "or a visible response."
        )
        if dialogue:
            state += f"\nUser: {dialogue}"
        if memory:
            state += (
                "\nMemory (prior experience; verify it against the current image "
                f"and telemetry):\n{memory}"
            )
        return state

    async def _receive(self, session, types):
        async for message in session.receive():
            update = message.session_resumption_update
            if update is not None and update.resumable and update.new_handle:
                self._session_handle = update.new_handle

            if message.go_away is not None:
                self._reconnect_requested = True

            usage = message.usage_metadata
            if usage is not None and usage.thoughts_token_count is not None:
                self.thought_token_count = usage.thoughts_token_count
            content = message.server_content
            if content is not None:
                turn = content.model_turn
                if turn is not None and turn.parts:
                    for part in turn.parts:
                        if not part.text:
                            continue
                        if getattr(part, "thought", False):
                            self._response_thoughts.append(part.text)
                        else:
                            self._response_parts.append(part.text)
                else:
                    transcript = content.output_transcription
                    if transcript is not None and transcript.text:
                        self._response_parts.append(transcript.text)
                if content.turn_complete:
                    self.turn_count += 1
                    self._finish_turn()
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
                async with self._send_lock:
                    await session.send_tool_response(function_responses=responses)
            if message.tool_call_cancellation is not None:
                self._cancel_action("Gemini cancelled it")

            if self._reconnect_requested:
                self._response_parts.clear()
                self._response_thoughts.clear()
                self._action_summary = ""
                return

    async def _execute(self, name: str, args: dict) -> dict:
        if name == "move":
            return await self._move(args)
        if name == "turn":
            return await self._turn(args)
        if name == "hover":
            cancelled = self._cancel_action("hover")
            self._movement = "stop"
            self._turn_direction = "stop"
            self._record_action("hover")
            return {
                "status": "hovering",
                "cancelled_action": cancelled or "none",
                "telemetry": _telemetry_text(self._telemetry),
                "scheduling": "INTERRUPT",
            }
        if name == "speak":
            message = str(args.get("message", "")).strip()
            if not message:
                return {"status": "rejected", "reason": "message is required"}
            self._record_action(f"speak: {message}")
            print(f"Companion: {message}", flush=True)
            return {"status": "spoken", "scheduling": "SILENT"}
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
        busy = self._busy_response()
        if busy is not None:
            return busy
        now = time.monotonic()
        action = ActiveAction(
            "move",
            direction,
            float(duration_s),
            now + float(duration_s),
        )
        self._active_action = action
        self._last_action_result = ""
        self._movement = direction
        self._turn_direction = "stop"
        self._record_action(f"started {self._action_label(action)}")
        return {
            "status": "started",
            "action": self._action_label(action),
            "movement_tools": "unavailable until this action completes",
            "telemetry": _telemetry_text(self._telemetry),
            "scheduling": "SILENT",
        }

    async def _turn(self, args: dict) -> dict:
        direction = str(args.get("direction", "")).strip().lower()
        angle_deg = args.get("angle_deg")
        if direction not in ("left", "right"):
            return {
                "status": "rejected",
                "reason": "direction must be left or right",
            }
        if (
            isinstance(angle_deg, bool)
            or not isinstance(angle_deg, (int, float))
            or not math.isfinite(angle_deg)
            or not MIN_TURN_DEG <= angle_deg <= MAX_TURN_DEG
        ):
            return {
                "status": "rejected",
                "reason": f"angle_deg must be {MIN_TURN_DEG} to {MAX_TURN_DEG}",
            }
        busy = self._busy_response()
        if busy is not None:
            return busy
        now = time.monotonic()
        angle_deg = float(angle_deg)
        action = ActiveAction(
            "turn",
            direction,
            angle_deg,
            now + angle_deg / TURN_RATE_DEG_S + ACTION_GRACE_S,
            self._telemetry.heading_rad
            if _finite(self._telemetry.heading_rad)
            else None,
        )
        self._active_action = action
        self._last_action_result = ""
        self._movement = "stop"
        self._turn_direction = direction
        self._record_action(f"started {self._action_label(action)}")
        return {
            "status": "started",
            "action": self._action_label(action),
            "movement_tools": "unavailable until this action completes",
            "telemetry": _telemetry_text(self._telemetry),
            "scheduling": "SILENT",
        }

    def _busy_response(self):
        self._refresh_action()
        action = self._active_action
        if action is None:
            return None
        return {
            "status": "unavailable",
            "reason": "a physical movement action is still in progress",
            "active_action": self._action_label(action),
            "phase": action.phase,
            "remaining_s": max(0.0, action.deadline_s - time.monotonic()),
            "movement_tools": "unavailable until the active action completes",
            "telemetry": _telemetry_text(self._telemetry),
            "scheduling": "SILENT",
        }

    def _action_label(self, action: Optional[ActiveAction] = None) -> str:
        action = action or self._active_action
        if action is None:
            return "none"
        if action.kind == "move":
            return f"move {action.direction} for {action.amount:.1f}s"
        return f"turn {action.direction} {action.amount:.0f} degrees"

    def _action_state_text(self) -> str:
        self._refresh_action()
        action = self._active_action
        if action is None:
            if self._last_action_result:
                return f"{self._last_action_result}; movement tools available"
            return "none; movement tools available"
        details = [
            f"{self._action_label(action)}; {action.phase}",
            f"remaining={max(0.0, action.deadline_s - time.monotonic()):.1f}s",
        ]
        actual = self._heading_change_deg(action)
        if actual is not None:
            details.append(f"observed heading change={actual:+.1f} degrees")
        details.append("move and turn tools unavailable until completion")
        details.append("hover may interrupt")
        return "; ".join(details)

    def _refresh_action(self):
        action = self._active_action
        if action is None:
            return
        now = time.monotonic()
        if action.kind == "turn":
            if action.start_heading_rad is None and _finite(self._telemetry.heading_rad):
                action.start_heading_rad = self._telemetry.heading_rad
            actual = self._heading_change_deg(action)
            if action.phase == "running" and actual is not None:
                requested_rad = math.radians(action.amount)
                progress_rad = math.radians(actual)
                if progress_rad >= requested_rad - HEADING_TOLERANCE_RAD:
                    action.phase = "settling"
                    action.stable_since_s = now
                    action.last_heading_rad = self._telemetry.heading_rad
                    action.deadline_s = max(
                        action.deadline_s,
                        now + ACTION_SETTLE_S,
                    )
                    self._movement = "stop"
                    self._turn_direction = "stop"
                    return
            if action.phase == "settling":
                heading = self._telemetry.heading_rad
                if _finite(heading) and action.last_heading_rad is not None:
                    change_rad = abs(_angle_delta_rad(action.last_heading_rad, heading))
                    if change_rad > HEADING_STABILITY_RAD:
                        action.stable_since_s = now
                    action.last_heading_rad = heading
                    if (
                        action.stable_since_s is not None
                        and now - action.stable_since_s >= ACTION_STABLE_S
                    ):
                        self._complete_action("completed", actual)
                        return
        if now >= action.deadline_s:
            actual = self._heading_change_deg(action)
            if action.kind == "turn" and actual is not None:
                requested = action.amount - math.degrees(HEADING_TOLERANCE_RAD)
                status = (
                    "completed"
                    if actual >= requested
                    else "timed out before target"
                )
                self._complete_action(status, actual)
            else:
                self._complete_action("completed")

    def _heading_change_deg(self, action: ActiveAction) -> Optional[float]:
        if action.start_heading_rad is None or not _finite(self._telemetry.heading_rad):
            return None
        change_rad = _angle_delta_rad(
            action.start_heading_rad,
            self._telemetry.heading_rad,
        )
        if action.direction == "left":
            change_rad = -change_rad
        return math.degrees(change_rad)

    def _complete_action(self, status: str, actual_heading_deg: Optional[float] = None):
        action = self._active_action
        if action is None:
            return
        result = f"{self._action_label(action)} {status}"
        if actual_heading_deg is not None:
            result += f"; observed heading change {actual_heading_deg:+.1f} degrees"
        self._last_action_result = result
        self._record_action(result)
        self._active_action = None
        self._movement = "stop"
        self._turn_direction = "stop"

    def _cancel_action(self, reason: str) -> str:
        action = self._active_action
        if action is None:
            return ""
        result = f"{self._action_label(action)} cancelled by {reason}"
        actual = self._heading_change_deg(action)
        if actual is not None:
            result += f"; observed heading change {actual:+.1f} degrees"
        self._last_action_result = result
        self._record_action(result)
        self._active_action = None
        self._movement = "stop"
        self._turn_direction = "stop"
        return self._action_label(action)

    def _record_action(self, action: str):
        action = " ".join(str(action).split())
        if action:
            self._action_summary = (
                f"{self._action_summary}; {action}"
                if self._action_summary
                else action
            )

    def _finish_turn(self):
        thought = _model_text(self._response_thoughts)
        response = _model_text(self._response_parts)
        action = self._action_summary or "none"
        self._response_thoughts.clear()
        self._response_parts.clear()
        self._action_summary = ""
        self.latest_thought = thought
        self.latest_response = response
        self.latest_action = action
        summary = thought or response or action
        now = time.monotonic()
        latency = (
            max(0.0, now - self._last_heartbeat_at_s)
            if self._last_heartbeat_at_s is not None
            else None
        )
        self.latest_observation = VisualObservation(
            timestamp_s=now,
            description=summary,
            movement=self._movement,
            confidence=1.0,
        )
        self.latest_decision = ConsciousDecision(
            intent="",
            dialogue=response,
            summary=summary,
        )
        self.latest_observation_duration_s = latency
        self.latest_decision_duration_s = latency
        self.observation_count += 1
        self.decision_count += 1
        if thought:
            self.thought_count += 1
            print(f"Gemini thought: {thought}", flush=True)
        if response:
            print(f"Gemini response: {response}", flush=True)
        if self.memory_store is not None:
            experience = _telemetry_text(self._telemetry)
            if action != "none":
                experience += f"; action={action}"
            if summary not in ("", "none") and action not in summary:
                experience += f"; summary={summary}"
            self.memory_store.remember(experience)


def _tools():
    """Return the high-level actions exposed to Gemini."""

    return [{"function_declarations": [
        {
            "name": "move",
            "description": "Move slowly in one body direction for a short time.",
            "behavior": "NON_BLOCKING",
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
            "name": "turn",
            "description": "Turn in place slowly to look in another direction.",
            "behavior": "NON_BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "direction": {
                        "type": "STRING",
                        "enum": ["left", "right"],
                    },
                    "angle_deg": {
                        "type": "NUMBER",
                        "description": "A turn from 15 through 90 degrees.",
                    },
                },
                "required": ["direction", "angle_deg"],
            },
        },
        {
            "name": "hover",
            "description": "Stop horizontal motion and hold position.",
            "behavior": "NON_BLOCKING",
            "parameters": {"type": "OBJECT", "properties": {}},
        },
        {
            "name": "speak",
            "description": "Say one short message to the nearby user.",
            "behavior": "NON_BLOCKING",
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
        "You are the high-level brain of an indoor DEXI 3 companion drone. Observe "
        "the latest forward camera image, user dialogue, telemetry, and previous "
        "outputs. Telemetry includes forward TOF distance, body velocity, and "
        "heading. You have no lidar or direct motor control; PX4 stabilizes the "
        "vehicle and holds altitude. "
        "Before every movement, analyze what you see and where the vehicle is, "
        "including its current heading and active action. After a movement, wait "
        "for it to complete and use the new image and telemetry before choosing "
        "the next movement. Think broadly, but move slowly and deliberately. "
        "You have optional tools for brief forward or lateral translation, turning "
        "in place, hovering, and speech. Choose freely whether to use one, use "
        "several, or use none; a heartbeat is not a requirement to act or speak. "
        "When you decide to act, call the matching tool; do not describe a tool "
        "call as JSON or in a code fence. Speak only when useful. "
        "The CM5 checks every physical action and may stop it. Physical movement "
        "is serialized: when the current state says a move or turn is in progress, "
        "do not call move or turn again. Keep observing and thinking until the "
        "state says the action completed. Do not use hover just to keep thinking "
        "or wait for another image; use it to stop for safety or an explicit need. "
        "Hover may interrupt the current action. "
        "Never request altitude, motors, attitude, position, or a long translation. "
        "Turn only through the turn tool. Use hover when stopping is appropriate or "
        "the scene or telemetry is unclear."
    )


def _jpeg(frame) -> bytes:
    """Turn one Gazebo BGR frame into the bounded JPEG sent to Gemini."""

    image = Image.fromarray(frame[:, :, ::-1])
    if image.width > MAX_IMAGE_WIDTH:
        image.thumbnail((MAX_IMAGE_WIDTH, image.height))
    output = BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def _model_text(parts) -> str:
    """Return useful model text without empty structured-output placeholders."""

    value = " ".join("".join(parts).split())
    if value.replace("```json", "").replace("```", "").strip() in {
        "{}",
        "[]",
        "null",
    }:
        return ""
    return value


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
                    command.yaw_rate_deg_s,
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
            f"heading_deg={_heading_number(telemetry.heading_rad)}",
        )
    )


def _number(value) -> str:
    """Format one finite telemetry value without pretending unknown is zero."""

    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.2f}"
    return "?"


def _heading_number(value) -> str:
    """Format one heading in degrees without hiding missing telemetry."""

    if _finite(value):
        return f"{math.degrees(value):.1f}"
    return "?"


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _angle_delta_rad(start: float, end: float) -> float:
    """Return the signed shortest heading change from start to end."""

    return (end - start + math.pi) % (2.0 * math.pi) - math.pi
