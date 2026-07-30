"""Verify the full camera-to-PX4 path in SITL."""

import asyncio
import argparse
import socket
import subprocess
import sys
import time

from mavsdk import System

from control.command_packet import CommandPacket
from control.mind import MacMind, Telemetry
from control.mind_runtime import MindRuntime
from control.udp_control import UdpControlService
from control.udp_sender import UdpCommandSender
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from onboard.ros2_bridge import LatestDistanceSensor
from onboard.velocity_forwarder import MavsdkVelocityForwarder
from sim.flight import RecordingForwarder, close_mavsdk, land, prepare, wait_for_offboard
from sim.fixed_brain import FixedLanguageModel
from sim.offboard_control import (
    DistanceMessage,
    COMMAND_DROPOUT_END_S,
    COMMAND_DROPOUT_START_S,
    FOLLOW_END_S,
    INVALID_DISTANCE_END_S,
    INVALID_DISTANCE_START_S,
    SECOND_FOLLOW_END_S,
    SECOND_FOLLOW_START_S,
    PROFILE_DURATION_S,
    SETPOINT_PERIOD_S,
    STALE_DISTANCE_END_S,
    STALE_DISTANCE_START_S,
    TARGET_LOST_END_S,
    TARGET_LOST_START_S,
    THIRD_FOLLOW_END_S,
    THIRD_FOLLOW_START_S,
    demo_obstacle_distance_m,
    demo_state,
)
from sim.video_loopback import image_sender_command
from sim.yolo_vision import YoloVisualModel
from vision.video_stream import (
    AsyncLatestFrameReader,
    GStreamerH264Receiver,
    H264StreamConfig,
    close_subprocess,
)


async def _stop_task(task, stop_event):
    """Stop a service task during cleanup."""

    stop_event.set()
    if task is not None and not task.done():
        try:
            await task
        except Exception:
            pass


async def run(image_path: str, expect_person: bool = False):
    video_config = H264StreamConfig(port=5014, width=640, height=480, framerate=15)
    video_receiver = GStreamerH264Receiver(video_config)
    frame_reader = AsyncLatestFrameReader(video_receiver)
    camera_process = None
    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    sender = None
    drone = System()
    control = None
    cm5_task = None
    mac_task = None
    mind_task = None
    telemetry_task = None
    offboard_task = None
    cm5_stop = asyncio.Event()
    mac_stop = asyncio.Event()
    mind_stop = asyncio.Event()
    distance_sensor = LatestDistanceSensor()
    flight_started_at = None
    armed = False
    offboard_started = False
    landed = False
    max_north_velocity = 0.0
    min_north_velocity = 0.0
    max_east_velocity = 0.0
    min_east_velocity = 0.0
    frames_received = 0
    video_fault_reported = False
    target_loss_reported = False
    command_dropout_reported = False
    malformed_packet_reported = False
    stale_packet_reported = False
    fault_socket = None

    def current_obstacle(timestamp_s):
        elapsed_s = max(0.0, timestamp_s - flight_started_at)
        if not STALE_DISTANCE_START_S <= elapsed_s < STALE_DISTANCE_END_S:
            distance_sensor.update(DistanceMessage(demo_obstacle_distance_m(elapsed_s)))
        return distance_sensor.read()

    def current_intent(timestamp_s):
        return demo_state(max(0.0, timestamp_s - flight_started_at))

    try:
        video_receiver.start()
        camera_process = subprocess.Popen(
            image_sender_command(video_config, image_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("RTP camera loopback started.")

        await prepare(drone)
        armed = True

        forwarder = MavsdkVelocityForwarder(drone)
        safe_commands = RecordingForwarder(forwarder)

        receiver.start()
        fault_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cm5_service = SafetyCommandService(
            receiver,
            safe_commands,
            tick_period_s=SETPOINT_PERIOD_S,
            obstacle_distance=distance_sensor.read,
        )
        cm5_service.start()
        cm5_task = asyncio.create_task(cm5_service.run(cm5_stop))
        sender = UdpCommandSender("127.0.0.1", receiver.port)

        class ScenarioSender:
            def start(self):
                sender.start()

            def send(self, command):
                nonlocal command_dropout_reported, malformed_packet_reported
                nonlocal stale_packet_reported
                elapsed_s = time.monotonic() - flight_started_at
                if COMMAND_DROPOUT_START_S <= elapsed_s < COMMAND_DROPOUT_END_S:
                    if not command_dropout_reported:
                        print("Command packet loss injected during following.")
                        command_dropout_reported = True
                    if not malformed_packet_reported:
                        fault_socket.sendto(
                            b"not-a-command",
                            ("127.0.0.1", receiver.port),
                        )
                        print("Malformed command packet injected during dropout.")
                        malformed_packet_reported = True
                    if not stale_packet_reported:
                        fault_socket.sendto(
                            CommandPacket(
                                0,
                                VelocityCommand(north_m_s=0.5),
                            ).encode(),
                            ("127.0.0.1", receiver.port),
                        )
                        print("Stale command packet injected during dropout.")
                        stale_packet_reported = True
                    return
                sender.send(command)

            def close(self):
                sender.close()

        mac_sender = ScenarioSender()

        control = MindRuntime(
            MacMind(
                YoloVisualModel(
                    model_path="yolov8n.pt",
                    frame_width_px=video_config.width,
                    target_height_px=video_config.height * 0.75,
                ),
                FixedLanguageModel(),
            )
        )
        flight_started_at = time.monotonic()
        mind_task = asyncio.create_task(
            control.think_loop(
                mind_stop,
                telemetry_provider=lambda: Telemetry(
                    obstacle_distance_m=distance_sensor.read()
                ),
            )
        )

        async def read_frame():
            nonlocal frames_received, target_loss_reported, video_fault_reported
            sample = await frame_reader.read()
            elapsed_s = (
                time.monotonic() - flight_started_at
                if flight_started_at is not None
                else 0.0
            )
            if TARGET_LOST_START_S <= elapsed_s < TARGET_LOST_END_S:
                if not target_loss_reported:
                    print("Camera frame loss injected during following.")
                    target_loss_reported = True
                return time.monotonic(), None
            if (
                flight_started_at is not None
                and time.monotonic() - flight_started_at >= PROFILE_DURATION_S - 3.0
            ):
                if not video_fault_reported:
                    print("Video stream fault injected during final hover.")
                    video_fault_reported = True
                return time.monotonic(), None
            if sample[1] is not None:
                frames_received += 1
            return sample

        mac_service = UdpControlService(
            control,
            mac_sender,
            read_frame,
            intent_provider=current_intent,
            obstacle_provider=current_obstacle,
            tick_period_s=SETPOINT_PERIOD_S,
        )
        mac_task = asyncio.create_task(mac_service.run(mac_stop))
        deadline = time.monotonic() + 5.0
        while len(safe_commands.commands) < 3 and time.monotonic() < deadline:
            await asyncio.sleep(SETPOINT_PERIOD_S)
        if len(safe_commands.commands) < 3:
            raise RuntimeError("CM5 did not forward consecutive priming setpoints")
        print("CM5 priming setpoints=verified.")
        await asyncio.sleep(SETPOINT_PERIOD_S * 4)
        await drone.offboard.start()
        offboard_started = True
        offboard_task = asyncio.create_task(wait_for_offboard(drone))
        flight_started_at = time.monotonic()
        print("Offboard started through full Mac/CM5 stack.")

        async def observe_velocity():
            nonlocal max_north_velocity, min_north_velocity
            nonlocal max_east_velocity, min_east_velocity
            async for velocity in drone.telemetry.velocity_ned():
                max_north_velocity = max(max_north_velocity, velocity.north_m_s)
                min_north_velocity = min(min_north_velocity, velocity.north_m_s)
                max_east_velocity = max(max_east_velocity, velocity.east_m_s)
                min_east_velocity = min(min_east_velocity, velocity.east_m_s)

        telemetry_task = asyncio.create_task(observe_velocity())
        await asyncio.sleep(PROFILE_DURATION_S + 2.0)
        await asyncio.wait_for(offboard_task, timeout=5.0)
        print("Offboard telemetry=verified through full Mac/CM5 stack.")
        if control.latest_decision is None:
            raise RuntimeError("full stack did not observe a conscious Mac decision")
        if control.latest_observation is None:
            raise RuntimeError("full stack did not observe a Mac visual observation")
        if not control.latest_decision.summary:
            raise RuntimeError("full stack did not retain a conscious visual summary")
        print("Conscious Mac decision=verified through full Mac/CM5 stack.")
        print("Conscious visual memory=verified through full Mac/CM5 stack.")
        print("Mac visual observation=verified through full Mac/CM5 stack.")
        try:
            await asyncio.wait_for(mac_task, timeout=2.0)
        except RuntimeError as error:
            if str(error) != "video stream stalled before control shutdown":
                raise
            print("Video stall=verified; Mac sent zero and stopped.")
        except asyncio.TimeoutError:
            raise RuntimeError("full stack did not observe the video-stall shutdown")
        else:
            raise RuntimeError("full stack did not observe the video-stall shutdown")
        mind_stop.set()
        await mind_task
        control.close()
        cm5_stop.set()
        await cm5_task
        if not safe_commands.commands or safe_commands.commands[-1][1] != VelocityCommand():
            raise RuntimeError("full stack did not observe CM5 shutdown zero")
        if frames_received == 0:
            raise RuntimeError("full stack did not receive a decoded RTP frame")
        print("CM5 shutdown zero=verified.")
        print(f"RTP frames received: {frames_received}.")

        def forward_count(start_s, end_s):
            return sum(
                command.north_m_s > 0.0
                for timestamp_s, command in safe_commands.commands
                if start_s <= timestamp_s - flight_started_at < end_s
            )

        print(
            "Forward commands by cycle: "
            f"{forward_count(0.0, FOLLOW_END_S)}, "
            f"{forward_count(SECOND_FOLLOW_START_S, SECOND_FOLLOW_END_S)}, "
            f"{forward_count(THIRD_FOLLOW_START_S, THIRD_FOLLOW_END_S)}."
        )
        if min_north_velocity >= -0.05:
            raise RuntimeError(
                f"full stack did not observe obstacle backoff: {min_north_velocity:.2f}m/s"
            )
        if expect_person:
            if max_north_velocity <= 0.02:
                raise RuntimeError(
                    f"full stack did not observe visual following: {max_north_velocity:.2f}m/s"
                )
            if max_east_velocity <= 0.02 and min_east_velocity >= -0.02:
                raise RuntimeError(
                    "full stack did not observe lateral visual tracking: "
                    f"{min_east_velocity:.2f}..{max_east_velocity:.2f}m/s"
                )
            print("Visual following and lateral tracking=verified.")
        elif any(
            timestamp_s >= flight_started_at
            and (
                command.north_m_s > 0.0
                or command.east_m_s != 0.0
                or command.down_m_s != 0.0
            )
            for timestamp_s, command in safe_commands.commands
        ):
            raise RuntimeError("full stack commanded motion without a detected person")
        else:
            print("No-person visual safe stop=verified.")

        expected_motion = (
            (lambda command: command.north_m_s > 0.0)
            if expect_person
            else (lambda command: command == VelocityCommand())
        )
        motion_name = "following" if expect_person else "safe stop without a person"

        def observed(start_s, end_s, predicate):
            return any(
                start_s <= timestamp_s - flight_started_at < end_s
                and predicate(command)
                for timestamp_s, command in safe_commands.commands
            )

        objectives = (
            (5.0, 7.0, lambda command: command == VelocityCommand(), "hover intent"),
            (
                SECOND_FOLLOW_START_S,
                SECOND_FOLLOW_END_S,
                expected_motion,
                f"{motion_name} after hover",
            ),
            (
                INVALID_DISTANCE_START_S,
                INVALID_DISTANCE_END_S,
                lambda command: command == VelocityCommand(),
                "invalid obstacle sensor fail-safe",
            ),
            (
                INVALID_DISTANCE_END_S + 0.1,
                SECOND_FOLLOW_END_S,
                expected_motion,
                f"{motion_name} recovery after invalid sensor",
            ),
            (
                STALE_DISTANCE_START_S + 0.2,
                STALE_DISTANCE_END_S,
                lambda command: command == VelocityCommand(),
                "stale obstacle sensor fail-safe",
            ),
            (
                STALE_DISTANCE_END_S + 0.1,
                SECOND_FOLLOW_END_S,
                expected_motion,
                f"{motion_name} recovery after stale obstacle sensor",
            ),
            (
                TARGET_LOST_START_S + 0.55,
                TARGET_LOST_END_S,
                lambda command: command == VelocityCommand(),
                "target-loss hold after vision expiry",
            ),
            (
                TARGET_LOST_END_S + 0.1,
                SECOND_FOLLOW_END_S,
                expected_motion,
                f"{motion_name} recovery after target loss",
            ),
            (
                COMMAND_DROPOUT_START_S + 0.2,
                COMMAND_DROPOUT_END_S,
                lambda command: command == VelocityCommand(),
                "CM5 command-dropout fail-safe",
            ),
            (
                COMMAND_DROPOUT_END_S + 0.1,
                THIRD_FOLLOW_END_S,
                expected_motion,
                f"{motion_name} recovery after command dropout",
            ),
            (
                SECOND_FOLLOW_END_S,
                THIRD_FOLLOW_START_S,
                lambda command: command == VelocityCommand(),
                "second hover intent",
            ),
            (
                THIRD_FOLLOW_START_S,
                THIRD_FOLLOW_END_S,
                expected_motion,
                f"{motion_name} after second hover",
            ),
            (
                THIRD_FOLLOW_END_S,
                PROFILE_DURATION_S,
                lambda command: command == VelocityCommand(),
                "final hover intent",
            ),
        )
        for start_s, end_s, predicate, objective in objectives:
            if not observed(start_s, end_s, predicate):
                raise RuntimeError(f"full stack did not observe {objective} at CM5")
            print(f"Mission objective passed: {objective} through CM5.")
        print("Repeated following and hover intent through CM5=verified.")
        print(f"Max observed north velocity: {max_north_velocity:.2f}m/s")
        print(f"Min observed north velocity: {min_north_velocity:.2f}m/s")
        print(
            "Observed east velocity range: "
            f"{min_east_velocity:.2f}..{max_east_velocity:.2f}m/s"
        )
        await drone.offboard.stop()
        offboard_started = False
        await land(drone)
        landed = True
    finally:
        await _stop_task(mac_task, mac_stop)
        await _stop_task(cm5_task, cm5_stop)
        mind_stop.set()
        if mind_task is not None and not mind_task.done():
            await mind_task
        if telemetry_task is not None:
            telemetry_task.cancel()
            await asyncio.gather(telemetry_task, return_exceptions=True)
        if offboard_task is not None and not offboard_task.done():
            offboard_task.cancel()
            await asyncio.gather(offboard_task, return_exceptions=True)
        if control is not None:
            control.close()
        if offboard_started:
            try:
                await drone.offboard.stop()
            except Exception:
                pass
        if armed and not landed:
            try:
                await land(drone)
            except Exception:
                pass
        receiver.close()
        video_receiver.close()
        if camera_process is not None:
            close_subprocess(camera_process)
        if sender is not None:
            sender.close()
        if fault_socket is not None:
            fault_socket.close()
        close_mavsdk(drone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the RTP image flight path")
    parser.add_argument("image_path")
    parser.add_argument(
        "--expect-person",
        action="store_true",
        help="require the image to produce following and lateral motion",
    )
    args = parser.parse_args()
    asyncio.run(run(args.image_path, args.expect_person))
