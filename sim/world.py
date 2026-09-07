"""A small fixed world for PX4 behavior checks.

It supplies repeatable targets and faults while the vehicle flies in Gazebo
through the real command path.
"""

import asyncio
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from mavsdk import System

from control.memory import CompanionMemory
from sim.mind import CompanionMind
from control.dialogue import DialogueInput
from sim.mind_runtime import (
    MAX_FRAME_GAP_S,
    MAX_MOVEMENT_AGE_S,
    MIN_MOVEMENT_CONFIDENCE,
    MindRuntime,
)
from control.safety_limits import (
    BACKOFF_SPEED_M_S,
    MAX_YAW_RATE_DEG_S,
    OBSTACLE_STOP_M,
)
from control.velocity import VelocityCommand, ned_to_body
from onboard.safety import LatestDistanceSensor
from sim.flight import (
    close_mavsdk,
    land,
    prepare,
    wait_for_offboard,
)
from sim.gazebo_camera import GazeboCamera
from sim.gazebo_depth import GazeboDepthRangefinder
from sim.gazebo_topic import GazeboPoseAnimator
from sim.offboard_control import (
    DistanceMessage,
    PROFILE_DURATION_S,
    SECOND_FOLLOW_END_S,
    SECOND_FOLLOW_START_S,
    SETPOINT_PERIOD_S,
    THIRD_FOLLOW_END_S,
    THIRD_FOLLOW_START_S,
)
from sim.safety_stack import SimulatedSafetyStack
from sim.world_fixture import (
    BRAIN_SHUTDOWN_START_S,
    CAMERA_STALL_END_S,
    CAMERA_STALL_START_S,
    CONSCIOUS_FAILURE_END_S,
    CONSCIOUS_FAILURE_START_S,
    CONTROL_PAUSE_END_S,
    CONTROL_PAUSE_START_S,
    DEFAULT_EXPLORATORY_INTENT,
    DROPOUT_END_S,
    DROPOUT_START_S,
    GEMINI_RECONNECT_END_S,
    HOVER_START_S,
    INVALID_COMMAND_END_S,
    INVALID_COMMAND_START_S,
    INVALID_SENSOR_END_S,
    INVALID_SENSOR_START_S,
    LINK_RECOVERY_END_S,
    LOW_CONFIDENCE_END_S,
    LOW_CONFIDENCE_START_S,
    MALFORMED_CONSCIOUS_END_S,
    MALFORMED_CONSCIOUS_START_S,
    MAX_EXPLORATORY_SPEED_M_S,
    MAX_HEADING_CHANGE_DEG,
    NO_OBSTACLE_DISTANCE_M,
    OBSTACLE_END_S,
    OBSTACLE_START_S,
    RECOVERY_END_S,
    STALE_SENSOR_END_S,
    STALE_SENSOR_START_S,
    SyntheticWorld,
    TARGET_LOST_END_S,
    TARGET_LOST_START_S,
    TARGET_RIGHT_END_S,
    TARGET_RIGHT_START_S,
    VELOCITY_TELEMETRY_END_S,
    VELOCITY_TELEMETRY_START_S,
    VISUAL_FAILURE_END_S,
    VISUAL_FAILURE_START_S,
    WorldLanguageModel,
    WorldVisualModel,
)
from sim.intent import parse_focus, parse_intent


CAMERA_WARMUP_S = 2.0
SNAPSHOT_AFTER_FRAMES = 10


async def run(
    exploratory: bool = False,
    faults: bool = False,
    camera: bool = False,
    world_name: Optional[str] = None,
    duration_s: float = PROFILE_DURATION_S,
    gemini: bool = False,
    initial_intent: str = DEFAULT_EXPLORATORY_INTENT,
    depth: bool = False,
    memory_path: Optional[Path] = None,
    snapshot_path: Optional[Path] = None,
    dialogue_request: Optional[str] = None,
    trace: bool = False,
    moving_person: bool = False,
):
    """Run one complete Gazebo world simulation."""

    if world_name is None:
        world_name = "objects" if exploratory and (camera or depth) else "default"
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("simulation duration must be positive")
    if not exploratory and duration_s < PROFILE_DURATION_S:
        raise ValueError("deterministic simulation duration cannot be shorter than its profile")
    if faults and not exploratory:
        raise ValueError("fault injection requires exploratory simulation")
    if not isinstance(initial_intent, str) or not initial_intent.strip():
        raise ValueError("initial situation or intent must be a non-empty string")
    initial_intent = initial_intent.strip()
    if camera and depth:
        raise ValueError("camera and depth modes cannot run together")
    if depth and not exploratory:
        raise ValueError("Gazebo depth mode requires exploratory simulation")
    if exploratory and world_name != "default" and not (camera or depth):
        raise ValueError(
            "a non-default exploratory world requires --camera or --depth"
        )
    if moving_person and not (
        exploratory and world_name == "objects" and (camera or depth)
    ):
        raise ValueError(
            "moving-person simulation requires exploratory objects camera or depth mode"
        )
    if faults and camera:
        raise ValueError("fault injection requires synthetic safety or depth mode")
    if gemini and not (exploratory and (camera or depth)):
        raise ValueError("Gemini simulation requires exploratory camera or depth mode")
    if memory_path is not None and not exploratory:
        raise ValueError("experience memory requires exploratory simulation")
    if snapshot_path is not None and not exploratory:
        raise ValueError("camera snapshot requires exploratory simulation")
    if snapshot_path is not None and not (camera or depth):
        raise ValueError("camera snapshot requires Gazebo camera or depth mode")
    if dialogue_request is not None and not exploratory:
        raise ValueError("dialogue request requires exploratory simulation")
    if dialogue_request is not None and not dialogue_request.strip():
        raise ValueError("dialogue request must not be empty")
    requested_intent = (
        parse_intent(dialogue_request)
        if dialogue_request is not None and not gemini
        else None
    )

    memory_store = CompanionMemory(memory_path) if memory_path is not None else None
    memory_before = memory_store.context() if memory_store is not None else ""

    sender = None
    drone = System()
    stack = None
    mind_task = None
    mind_stop = asyncio.Event()
    telemetry_task = None
    attitude_task = None
    offboard_task = None
    dialogue_input = DialogueInput(dialogue_request) if exploratory else None
    gazebo_camera = None
    gazebo_depth = None
    person_motion = None
    control = None
    applied_dialogue_intent = None
    brain_shutdown = False
    brain_reconnect_requested = False
    camera_stall_seen = False
    camera_recovered = False
    offboard_started = False
    armed = False
    landed = False
    distance_sensor = LatestDistanceSensor()
    north_velocity_m_s = None
    east_velocity_m_s = None
    down_velocity_m_s = None
    velocity_telemetry_seen = False
    vehicle_velocity_fresh = True
    current_heading_deg = None
    initial_yaw_deg = None
    max_heading_change_deg = 0.0

    async def stop_flight_services():
        """Stop flight services once, whether the run finished or failed."""

        nonlocal offboard_started
        if stack is not None:
            await stack.stop()
        mind_stop.set()
        if mind_task is not None and not mind_task.done():
            await mind_task
        if control is not None:
            control.close()
            if gemini:
                await control.wait_closed()
        for task in (telemetry_task, attitude_task, offboard_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if offboard_started:
            try:
                await drone.offboard.stop()
            except Exception:
                pass
            offboard_started = False
        if person_motion is not None:
            person_motion.close()

    try:
        world = SyntheticWorld(exploratory, faults)
        if camera or depth:
            camera_sensor = "IMX214" if depth else "camera"
            camera_model = "x500_depth" if depth else "x500_mono_cam"
            gazebo_camera = GazeboCamera(
                f"/world/{world_name}/model/{camera_model}_0/"
                f"link/camera_link/sensor/{camera_sensor}/image"
            )
            gazebo_camera.start()
        if depth:
            gazebo_depth = GazeboDepthRangefinder("/depth_camera")
            gazebo_depth.start()
        if gemini:
            from control.gemini_brain import GeminiRuntime

            control = GeminiRuntime(
                situation=initial_intent,
                memory=memory_store,
            )
            await control.start()

        heading_deg = await prepare(drone)
        armed = True
        current_heading_deg = heading_deg
        if gazebo_camera is not None:
            # The first rendered frames can arrive while Gazebo is still
            # spawning the vehicle. Give the brain a settled view.
            await asyncio.sleep(CAMERA_WARMUP_S)
        if moving_person:
            person_motion = GazeboPoseAnimator(
                world_name,
                "person",
                ((7.2, 3.0, 0.0), (7.2, 2.0, 0.0), (7.2, 4.0, 0.0)),
            )
            person_motion.start()

        async def observe_heading():
            nonlocal current_heading_deg, initial_yaw_deg, max_heading_change_deg
            async for attitude in drone.telemetry.attitude_euler():
                yaw_deg = attitude.yaw_deg
                if not math.isfinite(yaw_deg):
                    continue
                current_heading_deg = yaw_deg
                if initial_yaw_deg is None:
                    initial_yaw_deg = yaw_deg
                change_deg = (yaw_deg - initial_yaw_deg + 180.0) % 360.0 - 180.0
                max_heading_change_deg = max(
                    max_heading_change_deg,
                    abs(change_deg),
                )

        attitude_task = asyncio.create_task(observe_heading())

        def vehicle_velocity():
            if not vehicle_velocity_fresh:
                return (None, None, None, None)
            if (
                north_velocity_m_s is None
                or east_velocity_m_s is None
                or down_velocity_m_s is None
            ):
                return (None, None, None, None)
            heading_rad = math.radians(current_heading_deg)
            forward, right, down = ned_to_body(
                north_velocity_m_s,
                east_velocity_m_s,
                down_velocity_m_s,
                heading_rad,
            )
            return (forward, right, down, heading_rad)

        stack = SimulatedSafetyStack(
            drone,
            obstacle_distance=distance_sensor.read,
            velocity_provider=vehicle_velocity,
        )
        sender = stack.start()
        safe_commands = stack.forwarder
        started_at = time.monotonic()
        if control is None:
            visual_model = WorldVisualModel(
                world,
                started_at,
                synthetic_scene=not (camera or depth),
            )
            language_model = WorldLanguageModel(exploratory)
            language_model.started_at_s = started_at
            control = MindRuntime(
                CompanionMind(visual_model, language_model, memory=memory_store)
            )

        camera_frames = 0
        snapshot_saved = False
        depth_samples = 0
        valid_depth_samples = 0
        minimum_depth_distance = math.inf

        def brain_telemetry():
            nonlocal velocity_telemetry_seen
            telemetry = sender.telemetry()
            if any(
                value is not None
                for value in (
                    telemetry.forward_velocity_m_s,
                    telemetry.right_velocity_m_s,
                    telemetry.down_velocity_m_s,
                )
            ):
                velocity_telemetry_seen = True
            return telemetry

        def read_step(elapsed_s: float):
            nonlocal depth_samples, valid_depth_samples, minimum_depth_distance
            nonlocal vehicle_velocity_fresh
            step = world.step(elapsed_s)
            vehicle_velocity_fresh = step.velocity_fresh
            if gazebo_depth is not None:
                distance = math.nan
                sample = None
                if step.distance_fresh:
                    sample = gazebo_depth.latest()
                    if sample is not None:
                        distance_sensor.update(sample)
                        depth_samples += 1
                    distance = distance_sensor.read()
                    if sample is not None and math.isfinite(distance):
                        valid_depth_samples += 1
                        minimum_depth_distance = min(
                            minimum_depth_distance,
                            distance,
                        )
                    if math.isfinite(distance):
                        if step.obstacle_distance_m is None or not math.isfinite(
                            step.obstacle_distance_m
                        ):
                            distance = math.nan
                        else:
                            distance = min(distance, step.obstacle_distance_m)
                distance_override = (
                    not step.distance_fresh
                    or step.obstacle_distance_m != NO_OBSTACLE_DISTANCE_M
                )
                if sample is not None or distance_override:
                    distance_sensor.update(
                        DistanceMessage(distance, 0.0, 19.1)
                    )
                return replace(
                    step,
                    obstacle_distance_m=distance,
                    distance_fresh=math.isfinite(distance),
                )
            if camera:
                return replace(
                    step,
                    obstacle_distance_m=None,
                    distance_fresh=False,
                )
            if step.distance_fresh:
                distance_sensor.update(
                    DistanceMessage(
                        step.obstacle_distance_m
                        if step.obstacle_distance_m is not None
                        else math.nan
                    )
                )
            return replace(step, obstacle_distance_m=distance_sensor.read())

        def send_packet(timestamp_s: float, step, intent=None):
            nonlocal brain_shutdown, camera_frames, snapshot_saved
            nonlocal applied_dialogue_intent, brain_reconnect_requested
            nonlocal camera_stall_seen, camera_recovered
            frame = None
            if step.frame_fresh:
                frame = gazebo_camera.latest() if gazebo_camera else step
            if not step.frame_fresh:
                camera_stall_seen = True
            elif camera_stall_seen and frame is not None:
                camera_recovered = True
            if gazebo_camera is not None and frame is not None:
                camera_frames += 1
                if (
                    snapshot_path is not None
                    and not snapshot_saved
                    and camera_frames >= SNAPSHOT_AFTER_FRAMES
                ):
                    from PIL import Image

                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(frame[:, :, ::-1]).save(snapshot_path)
                    snapshot_saved = True
            if step.brain_shutdown and not brain_shutdown:
                control.close()
                mind_stop.set()
                brain_shutdown = True
            if (
                gemini
                and step.brain_reconnect
                and not brain_reconnect_requested
            ):
                print("Gemini session reconnect requested.", flush=True)
                control.request_reconnect()
                brain_reconnect_requested = True
            if gemini and dialogue_input is not None:
                message = dialogue_input.next()
                if message:
                    applied_dialogue_intent = parse_intent(message)
                    control.add_dialogue(message)
            telemetry = brain_telemetry()
            if gemini:
                command = control.tick(
                    frame=frame,
                    timestamp_s=timestamp_s,
                    telemetry=telemetry,
                )
            else:
                command = control.tick(
                    frame=frame,
                    timestamp_s=timestamp_s,
                    intent=intent,
                    telemetry=telemetry,
                )
            if step.transmit:
                sender.send(step.command_override or command)
            return command

        starting_intent = initial_intent if exploratory else "following"
        send_packet(
            time.monotonic(),
            read_step(0.0),
            intent=starting_intent,
        )
        deadline = time.monotonic() + 5.0
        while len(safe_commands.commands) < 3 and time.monotonic() < deadline:
            await asyncio.sleep(SETPOINT_PERIOD_S)
        if len(safe_commands.commands) < 3:
            raise RuntimeError("CM5 did not forward consecutive priming setpoints")
        print("CM5 priming setpoints=verified.")
        await drone.offboard.start()
        offboard_started = True
        offboard_task = asyncio.create_task(wait_for_offboard(drone))
        started_at = time.monotonic()
        if not gemini:
            visual_model.started_at_s = started_at
            language_model.started_at_s = started_at
        dialogue_provider = None
        if dialogue_input is not None and not gemini:
            def read_dialogue():
                nonlocal applied_dialogue_intent
                message = dialogue_input.next()
                if message:
                    applied_dialogue_intent = parse_intent(message)
                return message

            dialogue_provider = read_dialogue

        if not gemini:
            mind_task = asyncio.create_task(
                control.think_loop(
                    mind_stop,
                    telemetry_provider=brain_telemetry,
                    dialogue_provider=dialogue_provider,
                )
            )
        if dialogue_input:
            dialogue_input.start()
            print("Exploratory mission: observe the brain, then land and disarm.")
        else:
            print("Mission: follow, recover, handle faults, hover, land, and disarm.")

        max_forward_velocity = 0.0
        min_forward_velocity = 0.0
        max_right_velocity = 0.0
        min_right_velocity = 0.0
        max_down_velocity = 0.0
        min_down_velocity = 0.0
        last_traced_observation = 0
        last_traced_decision = 0
        last_observation_signature = None
        last_decision_signature = None
        last_traced_command = None
        last_traced_gemini_actions = 0
        last_traced_gemini_thought = ""
        last_traced_gemini_response = ""
        last_traced_gemini_action = "stop"
        requested_focus_answered = False

        async def observe_velocity():
            nonlocal max_forward_velocity, min_forward_velocity
            nonlocal max_right_velocity, min_right_velocity
            nonlocal max_down_velocity, min_down_velocity
            nonlocal north_velocity_m_s, east_velocity_m_s, down_velocity_m_s
            async for velocity in drone.telemetry.velocity_ned():
                north_velocity_m_s = velocity.north_m_s
                east_velocity_m_s = velocity.east_m_s
                down_velocity_m_s = velocity.down_m_s
                forward, right, down = ned_to_body(
                    velocity.north_m_s,
                    velocity.east_m_s,
                    velocity.down_m_s,
                    math.radians(current_heading_deg),
                )
                max_forward_velocity = max(max_forward_velocity, forward)
                min_forward_velocity = min(min_forward_velocity, forward)
                max_right_velocity = max(max_right_velocity, right)
                min_right_velocity = min(min_right_velocity, right)
                max_down_velocity = max(max_down_velocity, down)
                min_down_velocity = min(min_down_velocity, down)

        telemetry_task = asyncio.create_task(observe_velocity())
        reported = set()

        def event_for(elapsed_s, step):
            if TARGET_RIGHT_START_S <= elapsed_s < TARGET_RIGHT_END_S:
                return "target moved right"
            if TARGET_LOST_START_S <= elapsed_s < TARGET_LOST_END_S:
                return "target lost; holding"
            if (
                step.obstacle_distance_m is not None
                and step.obstacle_distance_m < OBSTACLE_STOP_M
            ):
                return "obstacle detected; backing off"
            if not step.distance_fresh:
                return "distance sensor dropout"
            if isinstance(step.obstacle_distance_m, float) and math.isnan(step.obstacle_distance_m):
                return "invalid obstacle reading"
            if not step.transmit:
                return "command link dropout"
            if not step.velocity_fresh:
                return "vehicle velocity telemetry dropout"
            if VISUAL_FAILURE_START_S <= elapsed_s < VISUAL_FAILURE_END_S:
                return "visual model failure; holding zero"
            if CONSCIOUS_FAILURE_START_S <= elapsed_s < CONSCIOUS_FAILURE_END_S:
                return "conscious model failure; holding zero"
            if (
                MALFORMED_CONSCIOUS_START_S
                <= elapsed_s
                < MALFORMED_CONSCIOUS_END_S
            ):
                return "malformed conscious decision; holding zero"
            if step.command_override is not None:
                return "out-of-bounds command"
            if LOW_CONFIDENCE_START_S <= elapsed_s < LOW_CONFIDENCE_END_S:
                return "low-confidence vision; holding"
            if SECOND_FOLLOW_START_S <= elapsed_s < SECOND_FOLLOW_END_S:
                return "intent changed back to following"
            if THIRD_FOLLOW_START_S <= elapsed_s < THIRD_FOLLOW_END_S:
                return "intent changed to following again"
            if step.brain_shutdown:
                return "Brain shutdown; holding zero"
            if elapsed_s >= HOVER_START_S:
                return "intent changed to hover"
            return None

        def trace_brain(elapsed_s, step, brain_command):
            nonlocal last_traced_observation, last_traced_decision
            nonlocal last_observation_signature, last_decision_signature
            nonlocal last_traced_command
            nonlocal last_traced_gemini_actions
            nonlocal last_traced_gemini_thought, last_traced_gemini_response
            nonlocal last_traced_gemini_action
            if not trace:
                return

            def clean(value):
                return " ".join(str(value).split()) or "none"

            if gemini:
                if (
                    control.latest_thought
                    and control.latest_thought != last_traced_gemini_thought
                ):
                    print(
                        f"[Gemini {elapsed_s:5.1f}s] thought="
                        f"{clean(control.latest_thought)}",
                        flush=True,
                    )
                    last_traced_gemini_thought = control.latest_thought
                if (
                    control.latest_response
                    and control.latest_response != last_traced_gemini_response
                ):
                    print(
                        f"[Gemini {elapsed_s:5.1f}s] response="
                        f"{clean(control.latest_response)}",
                        flush=True,
                    )
                    last_traced_gemini_response = control.latest_response
                if (
                    control.latest_action != last_traced_gemini_action
                    or control.action_count != last_traced_gemini_actions
                ):
                    print(
                        f"[Gemini {elapsed_s:5.1f}s] action="
                        f"{clean(control.latest_action)}; "
                        f"actions={control.action_count}",
                        flush=True,
                    )
                    last_traced_gemini_action = control.latest_action
                    last_traced_gemini_actions = control.action_count
                if control.turn_count != last_traced_decision:
                    if (
                        control.latest_thought
                        or control.latest_response
                        or control.latest_action not in ("none", "stop")
                    ):
                        print(
                            f"[Gemini {elapsed_s:5.1f}s] "
                            f"thought={clean(control.latest_thought)}; "
                            f"response={clean(control.latest_response)}; "
                            f"action={clean(control.latest_action)}; "
                            f"latency={control.latest_turn_duration_s:.2f}s",
                            flush=True,
                        )
                    last_traced_decision = control.turn_count
            else:
                if control.observation_count != last_traced_observation:
                    observation = control.latest_observation
                    if observation is not None:
                        signature = (
                            clean(observation.description),
                            clean(observation.focused_answer),
                            clean(observation.alternate_movement),
                            clean(observation.next_focus),
                            observation.movement,
                            round(observation.confidence, 2),
                        )
                        if signature != last_observation_signature:
                            print(
                                f"[VLM {elapsed_s:5.1f}s] "
                                f"{signature[0]}; answer={signature[1]}; "
                                f"alternate={signature[2] or 'none'}; "
                                f"next-focus={signature[3]}; movement={signature[4]}; "
                                f"confidence={signature[5]:.2f}; "
                                f"latency={control.latest_observation_duration_s:.2f}s",
                                flush=True,
                            )
                            last_observation_signature = signature
                    last_traced_observation = control.observation_count

                if control.decision_count != last_traced_decision:
                    decision = control.latest_decision
                    if decision is not None:
                        signature = (
                            clean(decision.intent),
                            decision.intent_changed,
                            clean(decision.focus),
                            clean(decision.summary),
                        )
                        if signature != last_decision_signature:
                            print(
                                f"[LLM {elapsed_s:5.1f}s] "
                                f"intent={signature[0]}; changed={signature[1]}; "
                                f"focus={signature[2]}; summary={signature[3]}; "
                                f"latency={control.latest_decision_duration_s:.2f}s",
                                flush=True,
                            )
                            last_decision_signature = signature
                    last_traced_decision = control.decision_count

            distance = step.obstacle_distance_m
            if not step.transmit:
                reason = "command link dropout"
            elif not step.velocity_fresh:
                reason = "CM5 has no fresh vehicle velocity"
            elif not step.distance_fresh or distance is None or not math.isfinite(distance):
                reason = "CM5 has no fresh distance reading"
            elif not step.frame_fresh:
                reason = "camera frame stall fail-safe"
            elif distance < OBSTACLE_STOP_M:
                reason = "CM5 obstacle protection"
            elif step.command_override is not None:
                reason = "CM5 rejected the injected invalid command"
            elif brain_command == VelocityCommand():
                if gemini:
                    reason = (
                        "waiting for the first Gemini action or turn"
                        if control.action_count == 0 and control.turn_count == 0
                        else "Gemini chose to hover or its short action expired"
                    )
                else:
                    decision = control.latest_decision
                    observation = control.latest_observation
                    if (
                        decision is not None
                        and parse_intent(decision.intent) == "hover"
                    ):
                        reason = "conscious intent is hover"
                    elif observation is None:
                        reason = (
                            "waiting for the first VLM observation"
                            if control.observation_count == 0
                            else "latest VLM observation failed"
                        )
                    elif (
                        control.latest_observation_age_s is None
                        or control.latest_observation_age_s > MAX_MOVEMENT_AGE_S
                    ):
                        reason = "VLM movement lease expired"
                    elif (
                        control.latest_frame_age_s is None
                        or control.latest_frame_age_s > MAX_FRAME_GAP_S
                    ):
                        reason = "camera frame lease expired"
                    elif observation.confidence < MIN_MOVEMENT_CONFIDENCE:
                        reason = "VLM confidence is below the movement threshold"
                    elif (
                        observation.focused_answer
                        and (
                            control.mind.awaiting_focus_answer
                            or parse_intent(control.mind.intent) is None
                        )
                    ):
                        reason = "visual focus confirmed"
                    elif observation.movement in ("stop", "hover"):
                        reason = f"VLM suggested {observation.movement}"
                    else:
                        reason = "Brain timing or intent refresh held zero"
            else:
                reason = "Brain suggested movement or turn"
            last_forwarded = (
                safe_commands.commands[-1][1]
                if safe_commands.commands
                else None
            )
            command_state = (brain_command, last_forwarded, reason)
            if command_state != last_traced_command:
                print(
                    f"[CMD {elapsed_s:5.1f}s] brain={brain_command}; "
                    f"cm5-last={last_forwarded or 'pending'}; reason={reason}",
                    flush=True,
                )
                last_traced_command = command_state

        try:
            while (elapsed := time.monotonic() - started_at) < duration_s:
                now = time.monotonic()
                step = read_step(elapsed)
                event = None if exploratory else event_for(elapsed, step)
                if event is not None and event not in reported:
                    print(event.capitalize() + ".")
                    reported.add(event)
                if not exploratory and CONTROL_PAUSE_START_S <= elapsed < CONTROL_PAUSE_END_S:
                    if "Brain control pause" not in reported:
                        print("Brain control pause; CM5 timeout holds zero.")
                        reported.add("Brain control pause")
                    await asyncio.sleep(CONTROL_PAUSE_END_S - elapsed)
                    now = time.monotonic()
                    step = read_step(now - started_at)
                    command = send_packet(now, step)
                    trace_brain(now - started_at, step, command)
                    continue
                command = send_packet(now, step)
                trace_brain(elapsed, step, command)
                if (
                    not gemini
                    and dialogue_request is not None
                    and control.latest_observation is not None
                    and control.latest_observation.focused_answer
                ):
                    requested_focus_answered = True
                await asyncio.sleep(SETPOINT_PERIOD_S)
            await asyncio.wait_for(offboard_task, timeout=5.0)
            print("Offboard telemetry=verified through synthetic world and CM5 safety.")
        finally:
            await stop_flight_services()

        commands = safe_commands.commands
        if not commands or commands[-1][1] != VelocityCommand():
            raise RuntimeError("SITL did not observe zero command on CM5 shutdown")
        if depth:
            minimum_forward_command = min(
                command.forward_m_s for _, command in commands
            )
            print(
                "Minimum CM5-forwarded forward command: "
                f"{minimum_forward_command:.2f}m/s"
            )

        if not gemini:
            decision = control.latest_decision
            if decision is None:
                raise RuntimeError("SITL did not observe a conscious brain decision")
            if not decision.summary:
                raise RuntimeError("SITL did not retain a conscious visual summary")
            if control.observation_count == 0:
                raise RuntimeError("SITL did not complete a visual observation")
            if control.decision_count == 0:
                raise RuntimeError("SITL did not complete a conscious thought")
        if not velocity_telemetry_seen:
            raise RuntimeError("SITL did not feed CM5 velocity telemetry to the brain")
        print("Brain velocity telemetry=verified.")
        if initial_yaw_deg is None:
            raise RuntimeError("SITL did not provide heading telemetry")
        max_yaw_rate_deg_s = max(
            abs(command.yaw_rate_deg_s) for _, command in commands
        )
        if max_yaw_rate_deg_s > MAX_YAW_RATE_DEG_S:
            raise RuntimeError(
                "CM5 forwarded an unsafe yaw rate: "
                f"{max_yaw_rate_deg_s:.1f} degrees/s"
            )
        print(
            "CM5 yaw-rate limit=verified: "
            f"maximum {max_yaw_rate_deg_s:.1f} degrees/s."
        )
        if not exploratory and max_heading_change_deg > MAX_HEADING_CHANGE_DEG:
            raise RuntimeError(
                "SITL heading changed unexpectedly: "
                f"{max_heading_change_deg:.1f} degrees"
            )
        if exploratory:
            print(
                "PX4 yaw control observed: "
                f"maximum change {max_heading_change_deg:.1f} degrees."
            )
        else:
            print(
                "PX4 heading hold=verified: "
                f"maximum change {max_heading_change_deg:.1f} degrees."
            )
        if requested_intent is not None:
            if applied_dialogue_intent != requested_intent:
                raise RuntimeError(
                    "SITL did not receive the scripted dialogue request: "
                    f"expected {requested_intent}, got {applied_dialogue_intent}"
                )
            print(f"Scripted dialogue=verified: intent={requested_intent}.")
            if requested_intent == "hover":
                if any(command != VelocityCommand() for _, command in commands):
                    raise RuntimeError("explicit hover dialogue commanded motion")
                print("Explicit hover dialogue stop=verified.")
        elif dialogue_request is not None:
            if gemini and control.dialogue_sent_count == 0:
                raise RuntimeError(
                    "SITL did not send the scripted Gemini dialogue"
                )
            print(
                "Scripted open-ended dialogue="
                f"{'sent to Gemini' if gemini else 'delivered to the conscious mind'}."
            )
        initial_focus = parse_focus(initial_intent)
        requested_focus = parse_focus(dialogue_request or "")
        if initial_focus and not requested_focus and not gemini:
            if decision.focus != initial_focus:
                raise RuntimeError(
                    "SITL did not honor the initial visual focus: "
                    f"expected {initial_focus}, got {decision.focus or 'none'}"
                )
            print(f"Initial visual focus=verified: {initial_focus}.")
        if requested_focus and gemini:
            print(f"Scripted visual request=delivered: {requested_focus}.")
        elif requested_focus:
            if decision.focus != requested_focus and not requested_focus_answered:
                raise RuntimeError(
                    "SITL did not honor the scripted visual focus: "
                    f"expected {requested_focus}, got {decision.focus or 'none'}"
                )
            print(
                "Scripted visual focus=verified: "
                f"{requested_focus} ({'answered' if requested_focus_answered else 'active'})."
            )
        if gemini:
            print(
                "Gemini ER 2 activity=verified."
                if not faults
                else "Gemini ER 2 fault handling=verified."
            )
        else:
            print(
                "Conscious brain decision=verified: "
                f"intent={decision.intent}, focus={decision.focus or 'none'}."
            )
            print("Conscious visual memory=verified.")
            print("Brain visual observation=verified.")
            print(f"Brain visual observations=verified ({control.observation_count}).")
        if gemini:
            print(
                "Gemini native thought summaries observed: "
                f"{control.thought_count}; thought tokens: {control.thought_token_count}."
            )
        else:
            print(f"Conscious thoughts=verified ({control.decision_count}).")
        if gemini:
            print("Gemini ER 2 brain=verified.")
            if control.video_frame_count < 2:
                raise RuntimeError("SITL did not stream Gemini video frames")
            print(
                "Gemini turns/actions observed: "
                f"{control.turn_count} turns, {control.action_count} actions."
            )
            print(f"Gemini live video frames=verified ({control.video_frame_count}).")
        if memory_store is not None:
            persisted_memory = CompanionMemory(memory_store.path).context()
            latest_memory = (
                persisted_memory.splitlines()[-1] if persisted_memory else ""
            )
            required_memory = ("obstacle=", "command=", "velocity=", "heading_deg=")
            if not gemini:
                required_memory = ("intent=",) + required_memory
            if not latest_memory or any(
                field not in latest_memory for field in required_memory
            ):
                raise RuntimeError(
                    "SITL did not persist a complete conscious experience"
                )
            if persisted_memory == memory_before:
                raise RuntimeError("SITL did not add a new conscious experience")
            print("Conscious experience memory=verified and reloadable.")

        def forward_count(start_s, end_s):
            return sum(
                command.forward_m_s > 0.0
                for timestamp_s, command in commands
                if start_s <= timestamp_s - started_at < end_s
            )

        if not exploratory:
            print(
                "Following commands by cycle: "
                f"{forward_count(0.0, 4.0)}, "
                f"{forward_count(SECOND_FOLLOW_START_S, SECOND_FOLLOW_END_S)}, "
                f"{forward_count(THIRD_FOLLOW_START_S, THIRD_FOLLOW_END_S)}."
            )

        def observed(start_s, end_s, predicate):
            return any(
                start_s <= timestamp_s - started_at < end_s and predicate(command)
                for timestamp_s, command in commands
            )

        if exploratory and faults:
            fault_checks = (
                (
                    OBSTACLE_START_S,
                    OBSTACLE_END_S,
                    lambda command: command.forward_m_s <= -BACKOFF_SPEED_M_S,
                    "CM5 obstacle backoff",
                ),
                (
                    INVALID_SENSOR_START_S,
                    INVALID_SENSOR_END_S,
                    lambda command: command == VelocityCommand(),
                    "invalid distance fail-safe",
                ),
                (
                    DROPOUT_START_S + 0.2,
                    DROPOUT_END_S,
                    lambda command: command == VelocityCommand(),
                    "command link dropout expiry",
                ),
                (
                    INVALID_COMMAND_START_S,
                    INVALID_COMMAND_END_S,
                    lambda command: command == VelocityCommand(),
                    "invalid command rejection",
                ),
                (
                    VELOCITY_TELEMETRY_START_S,
                    VELOCITY_TELEMETRY_END_S,
                    lambda command: command == VelocityCommand(),
                    "missing vehicle velocity fail-safe",
                ),
                (
                    STALE_SENSOR_START_S + 0.2,
                    STALE_SENSOR_END_S,
                    lambda command: command == VelocityCommand(),
                    "stale distance fail-safe",
                ),
            )
            for start_s, end_s, predicate, behavior in fault_checks:
                if duration_s < end_s:
                    continue
                if not observed(start_s, end_s, predicate):
                    raise RuntimeError(
                        f"exploratory fault run did not observe {behavior}"
                    )
                print(f"Exploratory fault passed: {behavior}.")
            if duration_s >= BRAIN_SHUTDOWN_START_S + 0.2:
                if not brain_shutdown or not observed(
                    BRAIN_SHUTDOWN_START_S,
                    duration_s,
                    lambda command: command == VelocityCommand(),
                ):
                    raise RuntimeError(
                        "exploratory fault run did not observe brain shutdown zero"
                    )
                print("Exploratory fault passed: brain shutdown zero.")
            if duration_s >= CAMERA_STALL_END_S + 0.2:
                if (
                    not camera_stall_seen
                    or not camera_recovered
                    or not observed(
                        CAMERA_STALL_START_S + 0.5,
                        CAMERA_STALL_END_S,
                        lambda command: command == VelocityCommand(),
                    )
                ):
                    raise RuntimeError(
                        "exploratory fault run did not observe camera stall fail-safe"
                    )
                print(
                    "Exploratory fault passed: camera stall fail-safe and recovery."
                )
            if gemini and duration_s >= GEMINI_RECONNECT_END_S + 0.2:
                if (
                    not brain_reconnect_requested
                    or control.session_reconnect_count == 0
                ):
                    raise RuntimeError(
                        "exploratory fault run did not observe Gemini session reconnect"
                    )
                print(
                    "Exploratory fault passed: Gemini session reconnect "
                    f"({control.session_reconnect_count})."
                )

        checks = () if exploratory else (
            (0.0, 1.0, lambda command: command.forward_m_s > 0.0, "forward following"),
            (
                TARGET_RIGHT_START_S,
                TARGET_RIGHT_END_S,
                lambda command: command.right_m_s > 0.0,
                "lateral target tracking",
            ),
            (
                CONTROL_PAUSE_START_S,
                CONTROL_PAUSE_END_S + 0.1,
                lambda command: command == VelocityCommand(),
                "Brain heartbeat pause fail-safe",
            ),
            (
                CONTROL_PAUSE_END_S + 0.1,
                TARGET_RIGHT_END_S,
                lambda command: command.right_m_s > 0.0,
                "Brain heartbeat recovery",
            ),
            (
                2.1,
                TARGET_LOST_END_S,
                lambda command: command == VelocityCommand(),
                "hold after target loss",
            ),
            (
                OBSTACLE_START_S,
                OBSTACLE_END_S,
                lambda command: command.forward_m_s < 0.0,
                "obstacle backoff",
            ),
            (
                OBSTACLE_END_S,
                RECOVERY_END_S,
                lambda command: command.right_m_s > 0.0,
                "visual detour after obstacle",
            ),
            (
                INVALID_SENSOR_START_S,
                INVALID_SENSOR_END_S,
                lambda command: command == VelocityCommand(),
                "invalid obstacle fail-safe",
            ),
            (
                LOW_CONFIDENCE_START_S,
                LOW_CONFIDENCE_END_S + 0.1,
                lambda command: command == VelocityCommand(),
                "low-confidence visual stop",
            ),
            (
                LOW_CONFIDENCE_END_S + 0.1,
                SECOND_FOLLOW_END_S,
                lambda command: command.forward_m_s > 0.0,
                "following recovery after low-confidence vision",
            ),
            (
                VISUAL_FAILURE_START_S,
                VISUAL_FAILURE_END_S + 0.1,
                lambda command: command == VelocityCommand(),
                "visual model failure fail-safe",
            ),
            (
                VISUAL_FAILURE_END_S + 0.1,
                LOW_CONFIDENCE_END_S + 0.2,
                lambda command: command.forward_m_s > 0.0,
                "visual model recovery",
            ),
            (
                STALE_SENSOR_START_S + 0.2,
                STALE_SENSOR_END_S,
                lambda command: command == VelocityCommand(),
                "stale obstacle fail-safe",
            ),
            (
                VELOCITY_TELEMETRY_START_S,
                VELOCITY_TELEMETRY_END_S,
                lambda command: command == VelocityCommand(),
                "missing vehicle velocity telemetry fail-safe",
            ),
            (
                VELOCITY_TELEMETRY_END_S + 0.1,
                SECOND_FOLLOW_END_S,
                lambda command: command.forward_m_s > 0.0,
                "following recovery after missing vehicle velocity telemetry",
            ),
            (
                STALE_SENSOR_END_S + 0.1,
                SECOND_FOLLOW_END_S,
                lambda command: command.forward_m_s > 0.0,
                "following recovery after stale obstacle sensor",
            ),
            (
                DROPOUT_START_S + 0.15,
                DROPOUT_END_S,
                lambda command: command == VelocityCommand(),
                "command-dropout expiry",
            ),
            (
                DROPOUT_END_S,
                LINK_RECOVERY_END_S,
                lambda command: command.forward_m_s > 0.0,
                "command-link recovery",
            ),
            (
                INVALID_COMMAND_START_S,
                INVALID_COMMAND_END_S,
                lambda command: command == VelocityCommand(),
                "invalid-command rejection",
            ),
            (
                HOVER_START_S,
                SECOND_FOLLOW_START_S,
                lambda command: command == VelocityCommand(),
                "first hover",
            ),
            (
                SECOND_FOLLOW_START_S,
                SECOND_FOLLOW_END_S,
                lambda command: command.forward_m_s > 0.0,
                "following after hover",
            ),
            (
                SECOND_FOLLOW_END_S,
                THIRD_FOLLOW_START_S,
                lambda command: command == VelocityCommand(),
                "second hover",
            ),
            (
                THIRD_FOLLOW_START_S,
                THIRD_FOLLOW_END_S,
                lambda command: command.forward_m_s > 0.0,
                "following after second hover",
            ),
            (
                CONSCIOUS_FAILURE_START_S,
                CONSCIOUS_FAILURE_END_S + 0.1,
                lambda command: command == VelocityCommand(),
                "conscious model failure fail-safe",
            ),
            (
                CONSCIOUS_FAILURE_END_S + 0.1,
                THIRD_FOLLOW_END_S,
                lambda command: command.forward_m_s > 0.0,
                "conscious model recovery",
            ),
            (
                MALFORMED_CONSCIOUS_START_S,
                MALFORMED_CONSCIOUS_END_S + 0.1,
                lambda command: command == VelocityCommand(),
                "malformed conscious decision fail-safe",
            ),
            (
                MALFORMED_CONSCIOUS_END_S + 0.1,
                THIRD_FOLLOW_END_S,
                lambda command: command.forward_m_s > 0.0,
                "malformed conscious decision recovery",
            ),
            (
                THIRD_FOLLOW_END_S,
                PROFILE_DURATION_S,
                lambda command: command == VelocityCommand(),
                "final hover",
            ),
        )
        for start_s, end_s, predicate, behavior in checks:
            if not observed(start_s, end_s, predicate):
                raise RuntimeError(f"SITL did not observe {behavior}")
            print(f"Mission objective passed: {behavior}.")
        if not exploratory:
            if not language_model.recovered_with_observation:
                raise RuntimeError(
                    "SITL conscious recovery did not retain visual context"
                )
            print("Conscious recovery retained visual context=verified.")
            if not language_model.malformed_conscious_seen:
                raise RuntimeError(
                    "SITL did not exercise malformed conscious decision handling"
                )
            print("Malformed conscious decision handling=verified.")
            if not brain_shutdown or not observed(
                BRAIN_SHUTDOWN_START_S,
                PROFILE_DURATION_S,
                lambda command: command == VelocityCommand(),
            ):
                raise RuntimeError("SITL did not observe zero after brain shutdown")
            print("Brain shutdown fail-safe=verified.")
            if max_forward_velocity <= 0.02:
                raise RuntimeError(
                    "SITL did not observe forward following: "
                    f"{max_forward_velocity:.2f}m/s"
                )
            if min_forward_velocity >= -0.05:
                raise RuntimeError(
                    "SITL did not observe obstacle backoff: "
                    f"{min_forward_velocity:.2f}m/s"
                )
            if max_right_velocity <= 0.02:
                raise RuntimeError(
                    "SITL did not observe lateral following: "
                    f"{max_right_velocity:.2f}m/s"
                )
        if exploratory and max(
            abs(max_forward_velocity),
            abs(min_forward_velocity),
            abs(max_right_velocity),
            abs(min_right_velocity),
            abs(max_down_velocity),
            abs(min_down_velocity),
        ) > MAX_EXPLORATORY_SPEED_M_S:
            raise RuntimeError(
                "exploratory flight exceeded its telemetry speed envelope: "
                f"forward={min_forward_velocity:.2f}..{max_forward_velocity:.2f}, "
                f"right={min_right_velocity:.2f}..{max_right_velocity:.2f}, "
                f"down={min_down_velocity:.2f}..{max_down_velocity:.2f}"
            )
        if camera or depth:
            if camera_frames == 0:
                raise RuntimeError("Gazebo did not provide a camera frame")
            print(f"Gazebo camera frames=verified ({camera_frames}).")
        if snapshot_path is not None:
            if not snapshot_saved:
                raise RuntimeError("Gazebo did not provide a frame snapshot")
            print(f"Gazebo frame snapshot=verified: {snapshot_path}")
        if camera:
            if any(command != VelocityCommand() for _, command in commands):
                raise RuntimeError(
                    "camera-only run forwarded motion without a TOF reading"
                )
            print("Gazebo camera-only safe stop=verified.")
        if depth:
            if not valid_depth_samples:
                raise RuntimeError("Gazebo did not provide a valid depth reading")
            print(
                "Gazebo depth samples=verified: "
                f"{valid_depth_samples} valid of {depth_samples}."
            )
            print(f"Minimum Gazebo depth distance: {minimum_depth_distance:.2f}m")
            if minimum_depth_distance < OBSTACLE_STOP_M:
                if not any(
                    command.forward_m_s <= -BACKOFF_SPEED_M_S
                    for _, command in commands
                ):
                    raise RuntimeError(
                        "CM5 did not back off after Gazebo depth reached the obstacle limit"
                    )
                print("CM5 Gazebo-depth obstacle backoff=verified.")
        print(f"Max observed forward velocity: {max_forward_velocity:.2f}m/s")
        print(f"Max observed right velocity: {max_right_velocity:.2f}m/s")
        print(f"Min observed forward velocity: {min_forward_velocity:.2f}m/s")
        print(
            "Observed right velocity range: "
            f"{min_right_velocity:.2f}..{max_right_velocity:.2f}m/s"
        )
        print(
            "Observed down velocity range: "
            f"{min_down_velocity:.2f}..{max_down_velocity:.2f}m/s"
        )
        await land(drone)
        landed = True
    finally:
        if gazebo_camera is not None:
            gazebo_camera.close()
        if gazebo_depth is not None:
            gazebo_depth.close()
        await stop_flight_services()
        if armed and not landed:
            try:
                await land(drone)
            except Exception:
                pass
        close_mavsdk(drone)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run the synthetic companion world")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument(
        "--faults",
        action="store_true",
        help="inject the normal safety and link faults into an exploratory run",
    )
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument(
        "--moving-person",
        action="store_true",
        help="move the visual mannequin through the objects world",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="use live Gemini ER 2 instead of the deterministic brain fixture",
    )
    parser.add_argument("--memory", type=Path, help="persist conscious experience across runs")
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="save the first Gazebo camera frame for visual inspection",
    )
    parser.add_argument(
        "--request",
        help="send one dialogue request automatically at the start of an exploratory run",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print meaningful brain observations, decisions, and command reasons",
    )
    parser.add_argument(
        "--world",
        help="Gazebo world name (exploratory runs default to objects)",
    )
    parser.add_argument("--duration", type=float, default=PROFILE_DURATION_S)
    parser.add_argument(
        "--intent",
        default=DEFAULT_EXPLORATORY_INTENT,
        help="initial situation for Gemini",
    )
    args = parser.parse_args()
    try:
        asyncio.run(
            run(
                exploratory=args.explore,
                faults=args.faults,
                camera=args.camera,
                depth=args.depth,
                world_name=args.world,
                duration_s=args.duration,
                gemini=args.gemini,
                initial_intent=args.intent,
                memory_path=args.memory,
                snapshot_path=args.snapshot,
                dialogue_request=args.request,
                trace=args.trace,
                moving_person=args.moving_person,
            )
        )
    except (RuntimeError, ValueError) as error:
        print(f"simulation: {error}", file=sys.stderr)
        raise SystemExit(1)
