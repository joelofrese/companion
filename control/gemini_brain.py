"""Run Gemini Robotics ER as the companion's streaming brain."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
import math
import os
import time
from typing import Optional

from PIL import Image

from control.memory import CompanionMemory
from control.safety_limits import OBSTACLE_STOP_M
from control.telemetry import Telemetry
from control.velocity import VelocityCommand, ned_to_body


DEFAULT_MODEL = "gemini-robotics-er-2-streaming-preview"
DEFAULT_SITUATION = "Explore the indoor surroundings autonomously."
# Keep each live control decision timely.
THINKING_LEVEL = "minimal"
# ER 2 Streaming accepts at most one JPEG per second.
VIDEO_PERIOD_S = 1.0
# Allow a slow ER 2 generation to finish.
# A silent turn is not useful for flight; recover before it feels stuck.
RESPONSE_TIMEOUT_S = 15.0
# ER 2 may take about half a minute to continue after a blocking tool result.
# The vehicle hovers while it thinks; a genuinely silent turn still recovers.
POST_ACTION_RESPONSE_TIMEOUT_S = RESPONSE_TIMEOUT_S
START_TIMEOUT_S = 20.0
INITIAL_CONNECT_RETRIES = 1
RECONNECT_DELAY_S = 1.0
MIN_MOVE_S = 0.2
MAX_MOVE_S = 2.0
DEFAULT_MOVE_DURATION_S = 1.0
MAX_FORWARD_SPEED_M_S = 0.25
MAX_RIGHT_SPEED_M_S = 0.20
MAX_VERTICAL_SPEED_M_S = 0.20
# The model asks for a relative angle; the runtime stops from measured heading.
MIN_TURN_DEG = 5.0
MAX_TURN_DEG = 90.0
# A turn without an angle is a small controller-like correction.
DEFAULT_TURN_DEG = 15.0
# Keep the yaw rate slow while making one visual correction useful.
TURN_RATE_DEG_S = 12.0
MIN_TURN_RATE_DEG_S = 1.5
TURN_SLOW_THRESHOLD_DEG = 10.0
# A new viewpoint requires translation after repeated view-only turns.
MAX_TURNS_WITHOUT_TRANSLATION = 2
MAX_IMAGE_WIDTH = 640
# PX4 may take longer than the commanded yaw rate to settle on a heading.
ACTION_GRACE_S = 5.0
ACTION_SETTLE_S = 1.0
MOVE_SETTLE_S = 0.5
ACTION_STABLE_S = 0.3
# Do not resume an action after safety has held it for too long.
ACTION_SAFETY_HOLD_S = 0.5
HEADING_STABILITY_RAD = math.radians(2.0)
HEADING_TOLERANCE_RAD = math.radians(2.0)
MAX_FRAME_AGE_S = 1.5
POST_ACTION_FRAME_TIMEOUT_S = VIDEO_PERIOD_S + 0.5
SPEECH_REPEAT_WINDOW_S = 5.0
HOLD_RECHECK_DELAY_S = 5.0


@dataclass
class ActiveAction:
    """One physical action that must finish before another can start."""

    kind: str
    direction: str
    duration_s: float
    deadline_s: float
    start_heading_rad: Optional[float] = None
    start_position_ned: Optional[tuple[float, float, float]] = None
    phase: str = "running"
    stable_since_s: Optional[float] = None
    last_heading_rad: Optional[float] = None
    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    down_m_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    last_update_s: Optional[float] = None
    last_sample_s: Optional[float] = None
    blocked_since_s: Optional[float] = None
    observed_forward_m: float = 0.0
    observed_right_m: float = 0.0
    observed_down_m: float = 0.0
    completion: Optional[dict] = None
    done: asyncio.Event = field(default_factory=asyncio.Event)


class GeminiRuntime:
    """Give one streaming Gemini session a deliberately small body."""

    def __init__(
        self,
        situation: str = DEFAULT_SITUATION,
        memory: Optional[CompanionMemory] = None,
        api_key: Optional[str] = None,
        include_thoughts: bool = False,
    ):
        if not isinstance(situation, str) or not situation.strip():
            raise ValueError("situation must be a non-empty string")
        self.situation = situation.strip()
        self.memory_store = memory
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.include_thoughts = include_thoughts
        self._latest_frame = None
        self._latest_frame_at_s: Optional[float] = None
        self._last_frame_sent_at_s: Optional[float] = None
        self._telemetry = Telemetry()
        self._dialogue = deque()
        self._last_dialogue = ""
        self._dialogue_in_flight: Optional[str] = None
        self._dialogue_send_complete = False
        self._dialogue_action_started = False
        self._hold_after_dialogue = False
        self._active_action: Optional[ActiveAction] = None
        self._initial_heading_rad: Optional[float] = None
        self._action_finished_at_s: Optional[float] = None
        self._hold_requested = False
        self._last_action_result = ""
        self._recent_action_results = deque(maxlen=3)
        self._turns_since_translation = 0
        self._last_spoken_message = ""
        self._last_spoken_at_s: Optional[float] = None
        self.latest_thought = ""
        self.latest_response = ""
        self.latest_action = "stop"
        self.latest_response_latency_s: Optional[float] = None
        self._last_turn_used_tool = False
        self.action_count = 0
        self.dialogue_sent_count = 0
        self.dialogue_count = 0
        self.video_frame_count = 0
        self.experience_count = 0
        self._response_parts = []
        self._response_thoughts = []
        self._actions = []
        self.thought_count = 0
        self.thought_token_count = 0
        self._memory_sent = False
        self._bootstrap_pending = True
        self._session_handle: Optional[str] = None
        self._session = None
        self.session_reconnect_count = 0
        self._reconnect_requested = False
        self._last_model_activity_s: Optional[float] = None
        self._response_in_flight = False
        self._decision_not_before_s: Optional[float] = None
        self._hold_tool_called = False
        self._speech_tool_called = False
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
        message = message.strip()
        self._last_dialogue = message
        self._decision_not_before_s = None
        self._dialogue_action_started = False
        self._hold_after_dialogue = _requests_hold_after(message)
        if _is_explicit_stop(message):
            self._hold_requested = True
            self._hold_after_dialogue = False
            self._cancel_action("explicit stop request")
        else:
            self._hold_requested = False
        if self._dialogue_in_flight is not None:
            self._dialogue.clear()
            self._dialogue_in_flight = None
            self._dialogue_send_complete = False
        self._dialogue.append(message)

    def request_reconnect(self):
        """Reconnect the live session while stopping any active movement."""

        if self._closed.is_set():
            return
        self._cancel_action("Gemini session reconnecting")
        self._reconnect_requested = True
        self._close_session()

    def tick(
        self,
        frame,
        timestamp_s: float,
        telemetry: Optional[Telemetry] = None,
    ) -> VelocityCommand:
        """Store fresh state and return the current bounded action."""

        if self._closed.is_set() or self._error is not None:
            return VelocityCommand()
        telemetry = telemetry or Telemetry()
        if frame is not None:
            self._latest_frame = frame
            self._latest_frame_at_s = time.monotonic()
            self._frame_ready.set()
        self._telemetry = telemetry
        if (
            self._initial_heading_rad is None
            and _finite(telemetry.heading_rad)
        ):
            self._initial_heading_rad = telemetry.heading_rad
        self._refresh_action()
        if not self._has_fresh_frame():
            self._cancel_action("camera frame stale")
            return VelocityCommand()
        action = self._active_action
        if (
            action is None
            or action.phase != "running"
            or self._action_is_blocked(action)
        ):
            return VelocityCommand()
        if action.kind == "move":
            forward_m_s = action.forward_m_s
            if (
                forward_m_s > 0.0
                and _obstacle_is_valid(self._telemetry.obstacle_distance_m)
                and self._telemetry.obstacle_distance_m <= OBSTACLE_STOP_M
            ):
                forward_m_s = 0.0
            return VelocityCommand(
                forward_m_s=forward_m_s,
                right_m_s=action.right_m_s,
                down_m_s=action.down_m_s,
                yaw_rate_deg_s=action.yaw_rate_deg_s,
            )
        if action.kind == "turn":
            yaw_rate = self._turn_rate(action)
            if action.direction == "left":
                yaw_rate = -yaw_rate
            return VelocityCommand(yaw_rate_deg_s=yaw_rate)
        return VelocityCommand()

    def close(self):
        """Stop movement and end the streaming session."""

        self._cancel_action("brain closed")
        self._closed.set()
        self._frame_ready.set()
        self._close_session()

    def _close_session(self):
        if self._session is not None:
            asyncio.create_task(self._session.close())

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
            initial_connect_attempts = 0
            while not self._closed.is_set():
                self._reconnect_requested = False
                self._session = None
                resuming_session = self._session_handle is not None
                if not resuming_session:
                    self._memory_sent = False
                    self._bootstrap_pending = True
                    self._dialogue_in_flight = None
                    self._dialogue_send_complete = False
                self._response_in_flight = False
                try:
                    config = types.LiveConnectConfig(
                        response_modalities=["TEXT"],
                        temperature=0.0,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=THINKING_LEVEL,
                            include_thoughts=self.include_thoughts
                        ),
                        tools=_tools(),
                        system_instruction=_system_instruction(),
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
                        model=DEFAULT_MODEL,
                        config=config,
                    ) as session:
                        self._session = session
                        if resuming_session:
                            # The resumable context already contains the user
                            # message sent before the previous connection ended.
                            self._discard_resumed_dialogue()
                        self._last_frame_sent_at_s = None
                        connected = True
                        self._ready.set()
                        await self._frame_ready.wait()
                        if self._closed.is_set():
                            return
                        heartbeat_task = asyncio.create_task(
                            self._heartbeat_loop(session, types)
                        )
                        try:
                            receive_task = None
                            response_started_s = None
                            while (
                                not self._closed.is_set()
                                and not self._reconnect_requested
                            ):
                                if receive_task is None:
                                    response_started_s = time.monotonic()
                                    self._last_model_activity_s = response_started_s
                                    receive_task = asyncio.create_task(
                                        self._receive(
                                            session,
                                            types,
                                            response_started_s,
                                        )
                                    )
                                done, _ = await asyncio.wait(
                                    (heartbeat_task, receive_task),
                                    timeout=VIDEO_PERIOD_S,
                                    return_when=asyncio.FIRST_COMPLETED,
                                )
                                if heartbeat_task in done:
                                    await heartbeat_task
                                    break
                                if receive_task in done:
                                    await receive_task
                                    receive_task = None
                                    response_started_s = None
                                    self._last_model_activity_s = None
                                    if self._reconnect_requested:
                                        break
                                elif (
                                    response_started_s is not None
                                    and self._response_is_stalled(response_started_s)
                                ):
                                    print(
                                        "Gemini response stalled; reconnecting the session.",
                                        flush=True,
                                    )
                                    receive_task.cancel()
                                    await asyncio.gather(
                                        receive_task,
                                        return_exceptions=True,
                                    )
                                    self._response_parts.clear()
                                    self._response_thoughts.clear()
                                    self._actions.clear()
                                    self._last_model_activity_s = None
                                    # A stalled generation can remain stalled when
                                    # resumed. Keep native resumption for ordinary
                                    # disconnects, but restart a silent session from
                                    # the situation and compact memory.
                                    self._session_handle = None
                                    self._memory_sent = False
                                    self._bootstrap_pending = True
                                    self._dialogue_in_flight = None
                                    self._dialogue_send_complete = False
                                    self._reconnect_requested = True
                                    break
                        finally:
                            tasks = [heartbeat_task]
                            if receive_task is not None:
                                tasks.append(receive_task)
                            for task in tasks:
                                task.cancel()
                            await asyncio.gather(
                                *tasks,
                                return_exceptions=True,
                            )
                            self._last_model_activity_s = None
                        if self._session_handle is None:
                            self._dialogue_in_flight = None
                            self._dialogue_send_complete = False
                        self._session = None
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._session = None
                    if self._closed.is_set():
                        return
                    if self._session_handle and _resume_rejected(error):
                        self._session_handle = None
                        self._memory_sent = False
                        self._bootstrap_pending = True
                        print(
                            "Gemini session resumption rejected; starting a fresh session.",
                            flush=True,
                        )
                    if not connected:
                        if initial_connect_attempts >= INITIAL_CONNECT_RETRIES:
                            raise error
                        initial_connect_attempts += 1
                        print(
                            f"Gemini initial connection failed; retrying: {error}",
                            flush=True,
                        )
                    else:
                        self._cancel_action("Gemini session reconnecting")
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

    async def _send_frame(self, session, types):
        """Send one current JPEG without waiting for a model decision."""

        frame = self._latest_frame
        if frame is None or not self._has_fresh_frame():
            return
        image_bytes = await asyncio.to_thread(_jpeg, frame)
        async with self._send_lock:
            now = time.monotonic()
            if (
                self._last_frame_sent_at_s is not None
                and now - self._last_frame_sent_at_s < VIDEO_PERIOD_S
            ):
                return
            await session.send_realtime_input(
                video=types.Blob(data=image_bytes, mime_type="image/jpeg")
            )
            self._last_frame_sent_at_s = now
        self.video_frame_count += 1

    async def _heartbeat_loop(self, session, types):
        """Continuously prompt Gemini with the newest frame and state."""

        while not self._closed.is_set():
            await self._heartbeat(session, types)
            try:
                await asyncio.wait_for(self._closed.wait(), timeout=VIDEO_PERIOD_S)
            except asyncio.TimeoutError:
                pass

    async def _heartbeat(self, session, types):
        """Send the current camera frame and state heartbeat."""

        await self._send_frame(session, types)
        # Let the native Live API turn continue after a tool response. Frames
        # keep streaming. New dialogue may interrupt a quiet turn when no
        # physical action is running.
        if self._response_in_flight:
            if (
                self._dialogue
                and self._dialogue_in_flight is None
                and self._active_action is None
            ):
                await self._send_dialogue(session, self._dialogue[0])
            return
        if self._hold_requested and not self._dialogue:
            return
        if (
            self._decision_not_before_s is not None
            and time.monotonic() < self._decision_not_before_s
            and not self._dialogue
        ):
            return
        # Video alone does not start a Live API reasoning turn. Send one text
        # heartbeat per turn, then let ER 2 finish or interrupt it itself.
        dialogue = (
            self._dialogue[0]
            if (
                self._dialogue
                and self._dialogue_in_flight is None
                and self._active_action is None
            )
            else ""
        )
        action_result = self._last_action_result
        self._response_in_flight = True
        self._hold_tool_called = False
        self._speech_tool_called = False
        if dialogue:
            await self._send_dialogue(session, dialogue)
        else:
            async with self._send_lock:
                try:
                    await session.send_realtime_input(
                        text=self._heartbeat_text("")
                    )
                except Exception:
                    self._response_in_flight = False
                    raise
        # Repeat a completed action once in the next heartbeat so the state is
        # easy to see even when the model did not close its turn.
        if action_result and self._last_action_result == action_result:
            self._last_action_result = ""
        if not self._memory_sent:
            self._memory_sent = True
        if self._bootstrap_pending:
            self._bootstrap_pending = False

    async def _send_dialogue(self, session, dialogue: str):
        """Send one queued dialogue message into the live session."""

        self._dialogue_in_flight = dialogue
        self._dialogue_send_complete = False
        try:
            async with self._send_lock:
                await session.send_realtime_input(
                    text=self._heartbeat_text(dialogue)
                )
        except Exception:
            self._response_in_flight = False
            if self._dialogue_in_flight == dialogue:
                self._dialogue_in_flight = None
                self._dialogue_send_complete = False
            raise
        if self._dialogue_in_flight == dialogue:
            self._dialogue_send_complete = True
            self.dialogue_sent_count += 1
            self._last_model_activity_s = time.monotonic()

    def _heartbeat_text(self, dialogue: str) -> str:
        memory = ""
        if not self._memory_sent:
            if self.memory_store is not None:
                memory = self.memory_store.context("experience=")
        parts = []
        if self._bootstrap_pending:
            parts.append(
                f"[START] Begin now. Treat this as the active situation, not a "
                f"request for permission: {self.situation} After inspecting the "
                "image, call exactly one useful tool; do not turn or move merely "
                "to begin."
            )
            if self._recent_action_results:
                parts.append(
                    "[RECENT ACTIONS] These measured actions happened before this "
                    "session. Use them as context with the current image and state:\n"
                    + "\n".join(self._recent_action_results)
                )
            if self._last_dialogue and not dialogue:
                parts.append(
                    "[ONGOING DIALOGUE] Continue the latest user request after "
                    "this fresh session reconnect:\n" + self._last_dialogue
                )
        camera = "fresh" if self._has_fresh_frame() else "stale"
        action_state = self._action_state_text()
        heading_from_initial = _relative_heading_number(
            self._initial_heading_rad,
            self._telemetry.heading_rad,
        )
        parts.append(
            f"[STATE] camera={camera}; "
            f"initial_heading_deg={_heading_number(self._initial_heading_rad)}; "
            f"heading_from_initial_deg={heading_from_initial}; "
            f"telemetry={_telemetry_text(self._telemetry)}; action={action_state}"
        )
        if dialogue:
            parts.append(f"[USER] {dialogue}")
        elif self._last_dialogue and self._bootstrap_pending:
            parts.append(
                "[CURRENT REQUEST] "
                f"{self._last_dialogue}\n"
                "If it is already fulfilled, hold position and wait for new dialogue."
            )
        if memory:
            parts.append(
                "[MEMORY] Prior experience; verify it against the current image "
                f"and telemetry:\n{memory}"
            )
        if self._active_action is not None:
            heartbeat = (
                "A physical action is still running. Observe its progress, but do "
                "not call move or turn until its measured result is returned."
            )
        elif self._last_action_result:
            heartbeat = (
                "The last physical action finished. Inspect its result and the "
                "newest image, then choose the next tool."
            )
        else:
            heartbeat = "Inspect the newest image and state, then choose the next tool."
        parts.append(
            f"[HEARTBEAT] {heartbeat} Choose move, turn, hover, or speak "
            "from the newest image and state. Continue the current situation and "
            "latest dialogue; wait when no useful safe change is clear."
        )
        if self.latest_response and not self._last_turn_used_tool:
            parts.append(
                "[NO EFFECT] The previous model turn was text only; it did not "
                "move or speak. Call a real tool directly if an action is needed."
            )
        return "\n".join(parts)

    def _has_fresh_frame(self) -> bool:
        return (
            self._latest_frame_at_s is not None
            and time.monotonic() - self._latest_frame_at_s <= MAX_FRAME_AGE_S
        )

    def _response_is_stalled(self, response_started_s: float) -> bool:
        """Return whether the current turn should be recovered."""

        if self._active_action is not None or self._hold_requested:
            return False
        now = time.monotonic()
        if (
            self._action_finished_at_s is not None
            and self._action_finished_at_s >= response_started_s
        ):
            return now - self._action_finished_at_s > POST_ACTION_RESPONSE_TIMEOUT_S
        return (
            now - (self._last_model_activity_s or response_started_s)
            > RESPONSE_TIMEOUT_S
        )

    async def _receive(self, session, types, response_started_s):
        async for message in session.receive():
            if (
                message.server_content is not None
                or message.tool_call is not None
                or message.tool_call_cancellation is not None
            ):
                self._last_model_activity_s = time.monotonic()
            turn_complete = False
            update = message.session_resumption_update
            if update is not None and update.resumable and update.new_handle:
                self._session_handle = update.new_handle

            if message.go_away is not None:
                self._cancel_action("Gemini session reconnecting")
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
                            self.latest_thought = _model_text(self._response_thoughts)
                        else:
                            self._response_parts.append(part.text)
                            self.latest_response = _model_text(self._response_parts)
                else:
                    transcript = content.output_transcription
                    if transcript is not None and transcript.text:
                        self._response_parts.append(transcript.text)
                turn_complete = bool(
                    content.turn_complete or content.generation_complete
                )
                if content.interrupted:
                    # Keep completed tool effects, but let the next heartbeat
                    # start a clean decision cycle.
                    self._acknowledge_dialogue()
                    self._response_parts.clear()
                    self._response_thoughts.clear()
                    self._actions.clear()
                    self._response_in_flight = False
                    return
            tool_call = message.tool_call
            if tool_call is not None:
                responses = []
                for call in tool_call.function_calls:
                    args = call.args or {}
                    result = await self._execute(call.name, args)
                    responses.append(
                        types.FunctionResponse(
                            name=call.name,
                            response=result,
                            id=call.id,
                        )
                    )
                if responses:
                    async with self._send_lock:
                        await session.send_tool_response(
                            function_responses=responses
                        )
                    self._last_model_activity_s = time.monotonic()
            if message.tool_call_cancellation is not None:
                self._cancel_action("Gemini cancelled it")

            if self._reconnect_requested:
                self._response_parts.clear()
                self._response_thoughts.clear()
                self._actions.clear()
                return
            if turn_complete:
                self._acknowledge_dialogue()
                self._finish_turn(response_started_s)
                self._response_in_flight = False
                return

    def _acknowledge_dialogue(self):
        """Remove a user message only after Gemini completes its turn."""

        if not self._dialogue_send_complete or not self._dialogue_in_flight:
            return
        if self._dialogue and self._dialogue[0] == self._dialogue_in_flight:
            self._dialogue.popleft()
            self.dialogue_count += 1
        self._dialogue_in_flight = None
        self._dialogue_send_complete = False

    def _discard_resumed_dialogue(self):
        """Forget a message already held by a resumed Gemini session."""

        if not self._dialogue_send_complete or not self._dialogue_in_flight:
            return
        if self._dialogue and self._dialogue[0] == self._dialogue_in_flight:
            self._dialogue.popleft()
        self._dialogue_in_flight = None
        self._dialogue_send_complete = False

    async def _execute(self, name: str, args: dict) -> dict:
        if name == "hover" and self._hold_tool_called:
            result = {
                "status": "unavailable",
                "reason": (
                    "the vehicle is already holding position; do not call another "
                    "hold tool in this turn; wait for the next heartbeat or dialogue"
                ),
                "telemetry": _telemetry_text(self._telemetry),
            }
        elif name == "move":
            result = await self._move(args)
        elif name == "turn":
            result = await self._turn(args)
        elif name == "hover":
            self._hold_tool_called = True
            result = self._hover()
        elif name == "speak":
            message = str(args.get("message", "")).strip()
            if not message:
                result = {"status": "rejected", "reason": "message is required"}
            elif self._speech_tool_called:
                result = {
                    "status": "unavailable",
                    "reason": "one speak call is enough for this turn; wait for the next turn",
                }
            elif (
                message == self._last_spoken_message
                and self._last_spoken_at_s is not None
                and time.monotonic() - self._last_spoken_at_s
                < SPEECH_REPEAT_WINDOW_S
            ):
                self._speech_tool_called = True
                self._record_action(f"speak already spoken: {message}")
                result = {
                    "status": "already_spoken",
                    "reason": "the same message was spoken moments ago",
                }
            elif (
                self._last_spoken_at_s is not None
                and time.monotonic() - self._last_spoken_at_s
                < SPEECH_REPEAT_WINDOW_S
                and not self._dialogue
                and self._dialogue_in_flight is None
            ):
                self._speech_tool_called = True
                self._record_action("speak unavailable: recent speech")
                result = {
                    "status": "unavailable",
                    "reason": "stay quiet briefly after a status message",
                }
            else:
                self._speech_tool_called = True
                self._last_spoken_message = message
                self._last_spoken_at_s = time.monotonic()
                self._record_action(f"speak: {message}")
                print(f"Companion: {message}", flush=True)
                self._remember_summary(message)
                result = {
                    "status": "spoken",
                }
        else:
            result = {"status": "rejected", "reason": "unknown tool"}
        if result.get("status") in ("already_hovering", "hovering", "spoken"):
            self.action_count += 1
        if result.get("status") in ("already_hovering", "hovering"):
            self._decision_not_before_s = (
                time.monotonic() + HOLD_RECHECK_DELAY_S
            )
        elif name in ("move", "turn", "speak"):
            self._decision_not_before_s = None
        if result.get("status") in ("rejected", "unavailable"):
            reason = str(result.get("reason", "")).strip()
            action = f"{name} {result['status']}"
            if reason:
                action += f": {reason}"
            self._record_action(action)
        return result

    def _hover(self) -> dict:
        if self._hold_after_dialogue and self._dialogue_action_started:
            self._hold_requested = True
            self._hold_after_dialogue = False
        if self._hold_requested:
            self._record_action("hover")
            return {
                "status": "hovering",
                "reason": "the explicit hold request remains active; wait for new dialogue",
                "heading_deg": _heading_value(self._telemetry.heading_rad),
                "telemetry": _telemetry_text(self._telemetry),
            }
        busy = self._busy_response()
        if busy is not None:
            return busy
        self._record_action("hover")
        return {
            "status": "already_hovering",
            "reason": "the vehicle is already holding position",
            "telemetry": _telemetry_text(self._telemetry),
        }

    async def _move(self, args: dict) -> dict:
        forward_m_s = _number_between(
            args,
            "forward_m_s",
            -MAX_FORWARD_SPEED_M_S,
            MAX_FORWARD_SPEED_M_S,
        )
        right_m_s = _number_between(
            args, "right_m_s", -MAX_RIGHT_SPEED_M_S, MAX_RIGHT_SPEED_M_S
        )
        up_m_s = 0.0
        if "up_m_s" in args:
            up_m_s = _number_between(
                args,
                "up_m_s",
                -MAX_VERTICAL_SPEED_M_S,
                MAX_VERTICAL_SPEED_M_S,
            )
        duration_s = DEFAULT_MOVE_DURATION_S
        if "duration_s" in args:
            duration_s = _number_between(args, "duration_s", MIN_MOVE_S, MAX_MOVE_S)
        yaw_rate_deg_s = 0.0
        if "yaw_rate_deg_s" in args:
            yaw_rate_deg_s = _number_between(
                args,
                "yaw_rate_deg_s",
                -TURN_RATE_DEG_S,
                TURN_RATE_DEG_S,
            )
        if (
            forward_m_s is None
            or right_m_s is None
            or up_m_s is None
            or duration_s is None
            or yaw_rate_deg_s is None
        ):
            return {
                "status": "rejected",
                "reason": (
                    "forward_m_s must be -"
                    f"{MAX_FORWARD_SPEED_M_S} to {MAX_FORWARD_SPEED_M_S}; "
                    "right_m_s must be "
                    f"-{MAX_RIGHT_SPEED_M_S} to {MAX_RIGHT_SPEED_M_S}; "
                    "up_m_s must be "
                    f"-{MAX_VERTICAL_SPEED_M_S} to {MAX_VERTICAL_SPEED_M_S}; "
                    "yaw_rate_deg_s must be "
                    f"-{TURN_RATE_DEG_S} to {TURN_RATE_DEG_S}; "
                    f"duration_s must be {MIN_MOVE_S} to {MAX_MOVE_S}"
                ),
            }
        if forward_m_s == 0.0 and right_m_s == 0.0 and up_m_s == 0.0:
            return {
                "status": "rejected",
                "reason": "at least one body-frame translation must be non-zero",
            }
        busy = self._busy_response()
        if busy is not None:
            return busy
        observation = self._observation_required_response()
        if observation is not None:
            return observation
        now = time.monotonic()
        action = ActiveAction(
            "move",
            _move_direction(forward_m_s, right_m_s, -up_m_s),
            duration_s,
            now + duration_s,
            start_heading_rad=(
                self._telemetry.heading_rad
                if yaw_rate_deg_s and _finite(self._telemetry.heading_rad)
                else None
            ),
            start_position_ned=_position_ned(self._telemetry),
            forward_m_s=forward_m_s,
            right_m_s=right_m_s,
            down_m_s=-up_m_s,
            yaw_rate_deg_s=yaw_rate_deg_s,
            last_update_s=now,
            last_sample_s=now,
        )
        self._active_action = action
        self._last_action_result = ""
        self._dialogue_action_started = True
        self._turns_since_translation = 0
        self.action_count += 1
        self._record_action(f"started {self._action_label(action)}")
        return await self._wait_for_action(action)

    async def _turn(self, args: dict) -> dict:
        direction = str(args.get("direction", "")).strip().lower()
        if direction not in ("left", "right"):
            return {
                "status": "rejected",
                "reason": "direction must be left or right",
            }
        angle_deg = DEFAULT_TURN_DEG
        if "angle_deg" in args:
            angle_deg = _number_between(
                args, "angle_deg", MIN_TURN_DEG, MAX_TURN_DEG
            )
            if angle_deg is None:
                return {
                    "status": "rejected",
                    "reason": (
                        "angle_deg must be "
                        f"{MIN_TURN_DEG} to {MAX_TURN_DEG} when provided"
                    ),
                }
        busy = self._busy_response()
        if busy is not None:
            return busy
        observation = self._observation_required_response()
        if observation is not None:
            return observation
        if self._turns_since_translation >= MAX_TURNS_WITHOUT_TRANSLATION:
            return {
                "status": "unavailable",
                "reason": (
                    "two turns changed the view without changing position; "
                    "use a short safe translation to create a new viewpoint "
                    "before turning again"
                ),
                "movement_tools": (
                    "move is available; turn is unavailable until translation"
                ),
                "telemetry": _telemetry_text(self._telemetry),
            }
        now = time.monotonic()
        angle_deg = float(angle_deg)
        duration_s = angle_deg / TURN_RATE_DEG_S
        action = ActiveAction(
            "turn",
            direction,
            duration_s,
            now + duration_s + ACTION_GRACE_S,
            start_heading_rad=(
                self._telemetry.heading_rad
                if _finite(self._telemetry.heading_rad)
                else None
            ),
            start_position_ned=_position_ned(self._telemetry),
            last_update_s=now,
        )
        self._active_action = action
        self._last_action_result = ""
        self._dialogue_action_started = True
        self._turns_since_translation += 1
        self.action_count += 1
        self._record_action(f"started {self._action_label(action)}")
        return await self._wait_for_action(action)

    async def _wait_for_action(self, action: ActiveAction) -> dict:
        """Return the measured result required by a blocking robotics tool."""

        await action.done.wait()
        finished_at_s = self._action_finished_at_s
        if finished_at_s is not None:
            try:
                await asyncio.wait_for(
                    self._wait_for_post_action_frame(finished_at_s),
                    timeout=POST_ACTION_FRAME_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                pass
        return action.completion or {
            "status": "cancelled",
            "action": self._action_label(action),
            "reason": "the action ended without a result",
        }

    async def _wait_for_post_action_frame(self, finished_at_s: float):
        """Wait until Gemini has received a camera frame after the action."""

        while not self._closed.is_set():
            if (
                self._last_frame_sent_at_s is not None
                and self._last_frame_sent_at_s > finished_at_s
            ):
                return
            await asyncio.sleep(0.05)

    def _busy_response(self):
        self._refresh_action()
        action = self._active_action
        if action is None:
            if self._dialogue and self._dialogue_in_flight is None:
                return {
                    "status": "unavailable",
                    "reason": (
                        "new dialogue is waiting to be processed; hold position "
                        "until it becomes the active turn"
                    ),
                    "movement_tools": "unavailable until the new dialogue is processed",
                    "telemetry": _telemetry_text(self._telemetry),
                }
            if self._hold_requested:
                return {
                    "status": "unavailable",
                    "reason": (
                        "an explicit hold request is active; wait for new dialogue "
                        "before moving again"
                    ),
                    "movement_tools": "unavailable until new dialogue",
                    "telemetry": _telemetry_text(self._telemetry),
                }
            return None
        reason = (
            "safety telemetry is currently holding the action; it will resume "
            "when movement is permitted"
            if self._action_is_blocked(action)
            else "a physical movement action is still in progress"
        )
        return {
            "status": "unavailable",
            "reason": reason,
            "active_action": self._action_label(action),
            "phase": action.phase,
            "remaining_s": max(0.0, action.deadline_s - time.monotonic()),
            "movement_tools": "unavailable until the active action completes",
            "telemetry": _telemetry_text(self._telemetry),
        }

    def _observation_required_response(self):
        if (
            self._action_finished_at_s is not None
            and (
                self._latest_frame_at_s is None
                or self._latest_frame_at_s <= self._action_finished_at_s
            )
        ):
            return {
                "status": "unavailable",
                "reason": (
                    "wait for a fresh camera frame after the previous physical "
                    "action before choosing another movement"
                ),
                "movement_tools": "unavailable until a fresh frame arrives",
                "telemetry": _telemetry_text(self._telemetry),
            }
        return None

    def _translation_text(self, action: ActiveAction) -> str:
        return (
            "requested translation "
            f"forward={action.forward_m_s * action.duration_s:+.2f}m "
            f"right={action.right_m_s * action.duration_s:+.2f}m; "
            f"up={-action.down_m_s * action.duration_s:+.2f}m; "
            "observed translation "
            f"forward={action.observed_forward_m:+.2f}m "
            f"right={action.observed_right_m:+.2f}m "
            f"up={-action.observed_down_m:+.2f}m"
        )

    def _position_delta(self, action: ActiveAction):
        if action.start_position_ned is None:
            return None
        current = _position_ned(self._telemetry)
        if current is None:
            return None
        heading = action.start_heading_rad
        if heading is None:
            heading = self._telemetry.heading_rad
        if not _finite(heading):
            return None
        north = current[0] - action.start_position_ned[0]
        east = current[1] - action.start_position_ned[1]
        down = current[2] - action.start_position_ned[2]
        return ned_to_body(north, east, down, heading)

    @staticmethod
    def _position_text(position_delta) -> str:
        return (
            "measured position change "
            f"forward={position_delta[0]:+.2f}m "
            f"right={position_delta[1]:+.2f}m "
            f"down={position_delta[2]:+.2f}m"
        )

    def _action_label(self, action: Optional[ActiveAction] = None) -> str:
        action = action or self._active_action
        if action is None:
            return "none"
        if action.kind == "move":
            label = (
                f"move forward={action.forward_m_s:+.2f}m/s "
                f"right={action.right_m_s:+.2f}m/s for {action.duration_s:.1f}s"
            )
            if action.yaw_rate_deg_s:
                label += f" yaw={action.yaw_rate_deg_s:+.1f}deg/s"
            if action.down_m_s:
                label += f" up={-action.down_m_s:+.2f}m/s"
            return label
        return f"turn {action.direction} {_turn_angle_deg(action):.0f}deg"

    def _action_state_text(self) -> str:
        self._refresh_action()
        action = self._active_action
        if action is None:
            if self._last_action_result:
                state = self._last_action_result
            else:
                state = "none"
            if self._turns_since_translation >= MAX_TURNS_WITHOUT_TRANSLATION:
                return (
                    f"{state}; turn unavailable until a short safe translation "
                    "creates a new viewpoint; move is available"
                )
            return f"{state}; movement tools available"
        details = [
            f"{self._action_label(action)}; {action.phase}",
            f"remaining={max(0.0, action.deadline_s - time.monotonic()):.1f}s",
        ]
        if action.kind == "move":
            details.append(self._translation_text(action))
        actual = self._heading_change_deg(action)
        if actual is not None:
            details.append(f"observed heading change={actual:+.1f} degrees")
        if action.kind == "turn":
            target = _target_heading_value(action)
            if target is not None:
                details.append(f"target heading={target:+.1f} degrees")
        if self._action_is_blocked(action):
            details.append("paused until safety telemetry permits movement")
        if self._turns_since_translation >= MAX_TURNS_WITHOUT_TRANSLATION:
            details.append(
                "two turns changed the view without changing position; "
                "translate safely before another turn"
            )
        details.append("move and turn tools unavailable until completion")
        details.append("hover may interrupt only for an explicit hold request")
        return "; ".join(details)

    def _refresh_action(self):
        action = self._active_action
        if action is None:
            return
        now = time.monotonic()
        self._record_translation(action, now)
        if self._action_is_blocked(action):
            if action.blocked_since_s is None:
                action.blocked_since_s = now
            elif now - action.blocked_since_s >= ACTION_SAFETY_HOLD_S:
                self._cancel_action("safety hold persisted")
                return
        else:
            action.blocked_since_s = None
        self._pause_action_if_blocked(action, now)
        if action.kind == "turn":
            if action.start_heading_rad is None and _finite(self._telemetry.heading_rad):
                action.start_heading_rad = self._telemetry.heading_rad
            actual = self._heading_change_deg(action)
            if action.phase == "running" and actual is not None:
                requested_rad = math.radians(_turn_angle_deg(action))
                progress_rad = math.radians(actual)
                if progress_rad >= _turn_completion_rad(requested_rad):
                    action.phase = "settling"
                    action.stable_since_s = now
                    action.last_heading_rad = self._telemetry.heading_rad
                    action.deadline_s = max(
                        action.deadline_s,
                        now + ACTION_SETTLE_S,
                    )
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
                        self._finish_action("completed", actual)
                        return
        if (
            action.kind == "move"
            and action.phase == "running"
            and now >= action.deadline_s
        ):
            # Stop commanding motion, then let telemetry catch up before
            # reporting how far the vehicle actually moved.
            action.phase = "settling"
            action.deadline_s = now + MOVE_SETTLE_S
            return
        if now >= action.deadline_s:
            actual = self._heading_change_deg(action)
            if action.kind == "turn" and actual is not None:
                completion_rad = _turn_completion_rad(
                    math.radians(_turn_angle_deg(action))
                )
                status = (
                    "completed"
                    if math.radians(actual) >= completion_rad
                    else "timed out before target"
                )
                self._finish_action(status, actual)
            else:
                self._finish_action("completed")

    def _record_translation(self, action: ActiveAction, now: float):
        if action.kind != "move":
            return
        if action.last_sample_s is None:
            action.last_sample_s = now
            return
        elapsed = max(0.0, now - action.last_sample_s)
        if not self._action_is_blocked(action):
            if _finite(self._telemetry.forward_velocity_m_s):
                action.observed_forward_m += (
                    self._telemetry.forward_velocity_m_s * elapsed
                )
            if _finite(self._telemetry.right_velocity_m_s):
                action.observed_right_m += self._telemetry.right_velocity_m_s * elapsed
            if _finite(self._telemetry.down_velocity_m_s):
                action.observed_down_m += self._telemetry.down_velocity_m_s * elapsed
        action.last_sample_s = now

    def _pause_action_if_blocked(self, action: ActiveAction, now: float):
        """Do not spend an action's time while safety state holds it still."""

        if (
            action.last_update_s is not None
            and action.phase == "running"
            and self._action_is_blocked(action)
        ):
            action.deadline_s += max(0.0, now - action.last_update_s)
        action.last_update_s = now

    def _action_is_blocked(self, action: ActiveAction) -> bool:
        if (
            not _obstacle_is_valid(self._telemetry.obstacle_distance_m)
            and action.kind != "turn"
        ):
            return True
        if action.kind != "turn" and not _move_is_allowed(
            action,
            self._telemetry.obstacle_distance_m
        ):
            return True
        if any(
            not _finite(value)
            for value in (
                self._telemetry.forward_velocity_m_s,
                self._telemetry.right_velocity_m_s,
                self._telemetry.down_velocity_m_s,
            )
        ):
            return True
        return action.kind == "turn" and not _finite(self._telemetry.heading_rad)

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

    def _turn_rate(self, action: ActiveAction) -> float:
        """Slow the turn as measured heading approaches its requested angle."""

        actual = self._heading_change_deg(action)
        if actual is None:
            return TURN_RATE_DEG_S
        remaining = max(0.0, _turn_angle_deg(action) - actual)
        if remaining >= TURN_SLOW_THRESHOLD_DEG:
            return TURN_RATE_DEG_S
        return max(
            MIN_TURN_RATE_DEG_S,
            TURN_RATE_DEG_S * remaining / TURN_SLOW_THRESHOLD_DEG,
        )

    def _cancel_action(self, reason: str) -> str:
        return self._finish_action(f"cancelled by {reason}")

    def _finish_action(
        self,
        status: str,
        actual_heading_deg: Optional[float] = None,
    ) -> str:
        """Record one physical action and its measured result."""

        action = self._active_action
        if action is None:
            return ""
        label = self._action_label(action)
        result = self._format_action_result(
            action,
            status,
            actual_heading_deg=actual_heading_deg,
        )
        action.completion = {
            "status": status,
            "action": label,
            "result": result,
            "telemetry": _telemetry_text(self._telemetry),
        }
        if action.kind == "turn":
            action.completion.update(
                requested_angle_deg=_turn_angle_deg(action),
                target_heading_deg=_target_heading_value(action),
                final_heading_deg=_heading_value(self._telemetry.heading_rad),
            )
            if actual_heading_deg is not None:
                action.completion["observed_angle_deg"] = actual_heading_deg
                action.completion["angle_error_deg"] = (
                    _turn_angle_deg(action) - actual_heading_deg
                )
        else:
            position_delta = self._position_delta(action)
            observed_translation = position_delta or (
                action.observed_forward_m,
                action.observed_right_m,
                action.observed_down_m,
            )
            action.completion["requested_translation_m"] = {
                "forward": action.forward_m_s * action.duration_s,
                "right": action.right_m_s * action.duration_s,
                "up": -action.down_m_s * action.duration_s,
            }
            action.completion["observed_translation_m"] = {
                "forward": observed_translation[0],
                "right": observed_translation[1],
                "up": -observed_translation[2],
            }
            if position_delta is not None:
                action.completion["position_change_m"] = {
                    "forward": position_delta[0],
                    "right": position_delta[1],
                    "down": position_delta[2],
                }
        self._save_action_result(result)
        action.done.set()
        return label

    def _format_action_result(
        self,
        action: ActiveAction,
        status: str,
        actual_heading_deg: Optional[float] = None,
    ) -> str:
        """Describe one action with the measurements available at its end."""

        if actual_heading_deg is None and action.start_heading_rad is not None:
            actual_heading_deg = self._heading_change_deg(action)
        result = f"{self._action_label(action)} {status}"
        if actual_heading_deg is not None:
            result += f"; observed heading change {actual_heading_deg:+.1f} degrees"
        if action.kind == "turn":
            target_heading = _target_heading_value(action)
            if target_heading is not None:
                result += f"; target heading {target_heading:+.1f} degrees"
            final_heading = _heading_value(self._telemetry.heading_rad)
            if final_heading is not None:
                result += f"; final heading {final_heading:+.1f} degrees"
        position_delta = self._position_delta(action)
        if position_delta is not None:
            result += f"; {self._position_text(position_delta)}"
        elif action.kind == "move":
            result += f"; {self._translation_text(action)}"
        return result

    def _save_action_result(self, result: str):
        """Publish one measured result to the live and persistent context."""

        action = self._active_action
        self._last_action_result = result
        self._recent_action_results.append(result)
        self._action_finished_at_s = time.monotonic()
        if (
            self.memory_store is not None
            and action.completion is not None
            and action.completion.get("status")
            in ("completed", "timed out before target")
        ):
            self.memory_store.remember(f"experience=action {result}")
            self.experience_count += 1
        self._record_action(result)
        self._active_action = None

    def _record_action(self, action: str):
        action = " ".join(str(action).split())
        if action:
            self._actions.append(action)
            self.latest_action = action

    def _remember_summary(self, summary: str):
        summary = _model_text([summary])
        if self.memory_store is None or not summary:
            return
        self.memory_store.remember(f"experience=summary {summary}")
        self.experience_count += 1

    def _finish_turn(self, response_started_s):
        thought = _model_text(self._response_thoughts)
        response = _model_text(self._response_parts)
        action = self._actions[-1] if self._actions else "none"
        self._response_thoughts.clear()
        self._response_parts.clear()
        self._actions.clear()
        self.latest_thought = thought
        self.latest_response = response
        self.latest_action = action
        self._last_turn_used_tool = action != "none"
        summary = thought or response
        if not summary:
            summary = action
        # This spans model output, blocking tool execution, and the fresh frame.
        self.latest_response_latency_s = (
            max(0.0, time.monotonic() - response_started_s)
            if summary and summary != "none"
            else None
        )
        if thought:
            self.thought_count += 1
            print(f"Gemini thought: {thought}", flush=True)
        if response:
            print(f"Gemini response: {response}", flush=True)
        if (
            self.memory_store is not None
            and summary not in ("", "none")
            and action != "none"
            and action not in summary
        ):
            self._remember_summary(summary)


def _tools():
    """Return the high-level actions exposed to Gemini."""

    return [{"function_declarations": [
        {
            "name": "move",
            "description": (
                "Move slowly in the body frame for a short pulse. Forward, right, "
                "and up are positive. Choose the direction from the newest image "
                "and telemetry. The range sensor looks forward only; do not move "
                "forward when that path is blocked. Inspect the measured result "
                "and a fresh image before another physical movement. A turn changes "
                "the view but not position; if an obstruction hides a target, "
                "translate to change the viewpoint instead of repeating turns."
            ),
            "behavior": "BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "forward_m_s": {
                        "type": "NUMBER",
                        "description": (
                            "Forward body velocity; negative is backward, from "
                            f"-{MAX_FORWARD_SPEED_M_S} through "
                            f"{MAX_FORWARD_SPEED_M_S} m/s."
                        ),
                        "minimum": -MAX_FORWARD_SPEED_M_S,
                        "maximum": MAX_FORWARD_SPEED_M_S,
                    },
                    "right_m_s": {
                        "type": "NUMBER",
                        "description": (
                            "Sideways body velocity: positive slides right and "
                            "negative slides left, from "
                            f"-{MAX_RIGHT_SPEED_M_S} through "
                            f"{MAX_RIGHT_SPEED_M_S} m/s."
                        ),
                        "minimum": -MAX_RIGHT_SPEED_M_S,
                        "maximum": MAX_RIGHT_SPEED_M_S,
                    },
                    "up_m_s": {
                        "type": "NUMBER",
                        "description": (
                            "Vertical body velocity; positive is up and negative is "
                            f"down, from -{MAX_VERTICAL_SPEED_M_S} through "
                            f"{MAX_VERTICAL_SPEED_M_S} m/s; use only for a short "
                            "clear adjustment, never as an altitude target."
                        ),
                        "minimum": -MAX_VERTICAL_SPEED_M_S,
                        "maximum": MAX_VERTICAL_SPEED_M_S,
                    },
                    "duration_s": {
                        "type": "NUMBER",
                        "description": (
                            f"Optional duration from {MIN_MOVE_S} through "
                            f"{MAX_MOVE_S} seconds; omit it for the default "
                            f"{DEFAULT_MOVE_DURATION_S}-second pulse. Use a useful "
                            "pulse in open space and a shorter one near an object; "
                            "inspect the measured result before moving again."
                        ),
                        "minimum": MIN_MOVE_S,
                        "maximum": MAX_MOVE_S,
                    },
                    "yaw_rate_deg_s": {
                        "type": "NUMBER",
                        "description": (
                            "Optional body yaw rate while translating; positive is "
                            f"right, from -{TURN_RATE_DEG_S} through "
                            f"{TURN_RATE_DEG_S} degrees per second."
                        ),
                        "minimum": -TURN_RATE_DEG_S,
                        "maximum": TURN_RATE_DEG_S,
                    },
                },
                "required": ["forward_m_s", "right_m_s"],
            },
        },
        {
            "name": "turn",
            "description": (
                "Turn slowly in place relative to the current nose. Choose the "
                "direction and relative angle from the newest image and heading. "
                "The controller stops from measured heading. Inspect the new image "
                "and measured result before another physical movement. Use move "
                "with yaw rate for a smooth translating turn."
            ),
            "behavior": "BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "direction": {
                        "type": "STRING",
                        "enum": ["left", "right"],
                        "description": (
                            "Relative to the current nose, not the room."
                        ),
                    },
                    "angle_deg": {
                        "type": "NUMBER",
                        "description": (
                            f"Optional relative heading change from {MIN_TURN_DEG:.0f} "
                            f"through {MAX_TURN_DEG:.0f} degrees. Omit it for a "
                            f"small {DEFAULT_TURN_DEG:.0f}-degree correction; choose "
                            "an explicit angle only when a larger change of view is "
                            "useful."
                        ),
                        "minimum": MIN_TURN_DEG,
                        "maximum": MAX_TURN_DEG,
                    },
                },
                "required": ["direction"],
            },
        },
        {
            "name": "hover",
            "description": (
                "Stop horizontal motion and hold position when waiting or when the "
                "scene is unclear. Let a normal move or turn finish; interrupt one "
                "only for an explicit hold request."
            ),
            "behavior": "BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {},
            },
        },
        {
            "name": "speak",
            "description": (
                "Call this function to say one short user-facing message; a text "
                "response is not spoken. Report physical outcomes only after "
                "observing them. Do not narrate routine movement or repeat a "
                "waiting status; after saying you are waiting, stay quiet until "
                "new dialogue or a meaningful change."
            ),
            "behavior": "NON_BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {"message": {"type": "STRING"}},
                "required": ["message"],
            },
        },
    ]}]


def _system_instruction() -> str:
    """State the control contract in plain language."""

    return """You are the high-level brain of an indoor DEXI 3 companion drone.
Use the newest camera image, forward TOF distance, velocity, local position,
heading, current action, dialogue, memory, and measured action results.

Use the declared functions directly. Describing a function call in text does
not move the drone; only `speak` produces speech. Treat the situation and
dialogue as ongoing context. Without a request, explore. When a request is
visibly fulfilled, hover and wait for new dialogue. An explicit hold stays
active until new dialogue; acknowledge it with `hover` once and do not move.
At startup, inspect the current image before moving or turning; do not make an
arbitrary scan turn when the current view already gives useful information.
When a request asks for movement followed by hovering, complete the movement
first and then call `hover` to remain there.

The camera faces forward. Image-left is negative right velocity and image-right
is positive. Heading increases clockwise. The TOF sensor only measures the path
ahead. Use fresh vision and valid telemetry; if the view is unclear, do not
invent an object or outcome.

Move in short, slow body-frame pulses. Forward, right, and up are positive; up
is only a brief adjustment. Never move forward when the path is blocked. A turn
changes the view but not position. If a target is hidden, translate to create a
new viewpoint instead of repeating turns. Make one small movement or turn, then
wait for its measured result and a fresh image before choosing another physical
action. Use measured heading and position to correct the next action. If a move
barely changes the position or view, reassess before repeating the same move. If
a requested object is already visible, do not turn just to search for it; keep
the view and approach only when the path is clear.

`move` and `turn` are blocking. After their result, continue the active request
from the new image and state. Use `hover` when no useful safe change is clear.
Use at most one `speak` call in a turn and combine the message instead of
sending several. After saying that you are waiting, stay quiet until new
dialogue or a meaningful change. The CM5 limits every command; never send
motors, attitude, altitude, or absolute-position commands.""".strip()


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
    empty_structured_value = value.casefold().replace("`", "").replace(" ", "")
    if empty_structured_value in {"{}", "json{}"}:
        return ""
    if (
        value.startswith("[START]")
        or value.startswith("[STATE]")
        or value.casefold().startswith("# no action chosen")
    ):
        return ""
    if value.casefold().rstrip(".!?") in {
        "none",
        "no action",
        "no tool call necessary",
        "no tool call is necessary",
    }:
        return ""
    if not any(character.isalnum() for character in value):
        return ""
    return value


def _telemetry_text(telemetry: Telemetry) -> str:
    """Return the small telemetry record useful to the model and memory."""

    command = telemetry.last_command or VelocityCommand()
    return "; ".join(
        (
            f"forward_path={_path_status(telemetry.obstacle_distance_m)}",
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
            "position_ned_m=" + ",".join(
                _number(value)
                for value in (
                    telemetry.position_north_m,
                    telemetry.position_east_m,
                    telemetry.position_down_m,
                )
            ),
        )
    )


def _position_ned(telemetry: Telemetry):
    """Return a complete local position, or none when it is unavailable."""

    position = (
        telemetry.position_north_m,
        telemetry.position_east_m,
        telemetry.position_down_m,
    )
    return position if all(_finite(value) for value in position) else None


def _path_status(distance_m: Optional[float]) -> str:
    """Describe only the forward TOF reading in plain language."""

    if not _obstacle_is_valid(distance_m):
        return "unknown"
    return "clear" if distance_m > OBSTACLE_STOP_M else "blocked"


def _number(value) -> str:
    """Format one finite telemetry value without pretending unknown is zero."""

    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.2f}"
    return "?"


def _number_between(args: dict, name: str, minimum: float, maximum: float):
    """Return one finite numeric argument inside its allowed range."""

    value = args.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        return None
    return float(value)


def _move_direction(forward_m_s: float, right_m_s: float, down_m_s: float) -> str:
    """Give a movement a compact label for trace output."""

    if forward_m_s and right_m_s:
        return "move"
    if forward_m_s:
        return "forward" if forward_m_s > 0.0 else "backward"
    if right_m_s:
        return "left" if right_m_s < 0.0 else "right"
    return "down" if down_m_s > 0.0 else "up"


def _move_is_allowed(action: ActiveAction, distance_m: Optional[float]) -> bool:
    """Allow non-forward motion when only the forward path is blocked."""

    return _obstacle_is_valid(distance_m) and (
        action.forward_m_s <= 0.0
        or distance_m > OBSTACLE_STOP_M
        or any(
            value != 0.0
            for value in (
                action.right_m_s,
                action.down_m_s,
                action.yaw_rate_deg_s,
            )
        )
    )


def _obstacle_is_valid(distance_m: Optional[float]) -> bool:
    """Return whether a forward range reading is usable."""

    return _finite(distance_m) and distance_m >= 0.0


def _heading_number(value) -> str:
    """Format one heading in degrees without hiding missing telemetry."""

    if _finite(value):
        return f"{math.degrees(value):.1f}"
    return "?"


def _heading_value(value):
    """Return a numeric heading for structured tool feedback."""

    return math.degrees(value) if _finite(value) else None


def _relative_heading_number(start, current) -> str:
    """Format the signed heading change from a reference heading."""

    if not _finite(start) or not _finite(current):
        return "?"
    return f"{math.degrees(_angle_delta_rad(start, current)):.1f}"


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_explicit_stop(message: str) -> bool:
    """Recognize dialogue that explicitly asks the drone to hold position."""

    message = _normalize_dialogue(message)
    for prefix in ("please ", "can you ", "could you ", "would you "):
        if message.startswith(prefix):
            message = message[len(prefix):]
            break
    for ending in (" now", " immediately"):
        if message.endswith(ending):
            message = message[:-len(ending)]
            break
    if message in {
        "stop",
        "stop moving",
        "hover",
        "hold position",
        "cancel",
        "cancel movement",
    }:
        return True
    if message.startswith(("stop ", "hover ", "hold position ", "cancel ")):
        return True
    for phrase in ("do not move", "don't move", "do not turn", "don't turn"):
        index = message.find(phrase)
        if index < 0:
            continue
        remainder = message[index + len(phrase):].lstrip()
        if not remainder.startswith(
            ("forward", "backward", "left", "right", "up", "down")
        ):
            return True
    if any(phrase in message for phrase in ("stay still", "remain still")):
        return True
    return False


def _requests_hold_after(message: str) -> bool:
    """Return whether a request asks for a hold after its other work."""

    message = _normalize_dialogue(message)
    return any(
        phrase in message
        for phrase in (
            " and hover",
            " then hover",
            " and hold position",
            " then hold position",
            " and wait",
            " then wait",
        )
    )


def _normalize_dialogue(message: str) -> str:
    """Normalize spoken dialogue for simple phrase matching."""

    for punctuation in ",.!?;:":
        message = message.replace(punctuation, " ")
    return " ".join(message.casefold().split())


def _resume_rejected(error: Exception) -> bool:
    text = str(error).casefold()
    return "1007" in text or "invalid frame payload" in text


def _angle_delta_rad(start: float, end: float) -> float:
    """Return the signed shortest heading change from start to end."""

    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def _turn_completion_rad(requested_rad: float) -> float:
    """Require meaningful progress even for the smallest allowed turn."""

    return max(requested_rad * 0.5, requested_rad - HEADING_TOLERANCE_RAD)


def _target_heading_value(action: ActiveAction):
    """Return the requested turn's target heading in the vehicle's angle range."""

    if action.start_heading_rad is None or not _finite(action.start_heading_rad):
        return None
    change_rad = math.radians(_turn_angle_deg(action))
    if action.direction == "left":
        change_rad = -change_rad
    target_rad = (action.start_heading_rad + change_rad + math.pi) % (
        2.0 * math.pi
    ) - math.pi
    return math.degrees(target_rad)


def _turn_angle_deg(action: ActiveAction) -> float:
    """Return the requested relative heading change for one turn."""

    return action.duration_s * TURN_RATE_DEG_S
