"""Verify RTP vision, Mac UDP control, CM5 safety, and PX4 SITL together."""

import asyncio
import subprocess
import sys
import time

from mavsdk import System

from control.following import FollowConfig, VisualFollower
from control.runtime import CompanionRuntime
from control.state_machine import ReactiveController
from control.udp_control import UdpControlService
from control.udp_sender import UdpCommandSender
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from onboard.ros2_bridge import LatestDistanceSensor
from onboard.velocity_forwarder import MavsdkVelocityForwarder
from sim.flight import close_mavsdk, land, prepare, wait_for_offboard
from sim.offboard_control import (
    DistanceMessage,
    FOLLOW_END_S,
    INVALID_DISTANCE_END_S,
    INVALID_DISTANCE_START_S,
    SECOND_FOLLOW_END_S,
    SECOND_FOLLOW_START_S,
    PROFILE_DURATION_S,
    SETPOINT_PERIOD_S,
    TARGET_LOST_END_S,
    TARGET_LOST_START_S,
    THIRD_FOLLOW_END_S,
    THIRD_FOLLOW_START_S,
    demo_obstacle_distance_m,
    demo_state,
)
from sim.video_loopback import image_sender_command
from vision.latest import LatestVisionPipeline
from vision.person_detector import YoloPersonDetector
from vision.pipeline import PersonVisionPipeline
from vision.video_stream import AsyncLatestFrameReader, GStreamerH264Receiver, H264StreamConfig, close_subprocess


async def _stop_task(task, stop_event):
    """Request service shutdown and suppress only cleanup-time task errors."""

    stop_event.set()
    if task is not None and not task.done():
        try:
            await task
        except Exception:
            pass


async def run(image_path: str):
    video_config = H264StreamConfig(port=5014, width=640, height=480, framerate=15)
    video_receiver = GStreamerH264Receiver(video_config)
    frame_reader = AsyncLatestFrameReader(video_receiver)
    camera_process = None
    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    sender = None
    drone = System()
    vision = None
    cm5_task = None
    mac_task = None
    telemetry_task = None
    offboard_task = None
    cm5_stop = asyncio.Event()
    mac_stop = asyncio.Event()
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

    def current_obstacle(timestamp_s):
        elapsed_s = max(0.0, timestamp_s - flight_started_at)
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
        safe_commands = []

        class ObservingForwarder:
            async def send(self, command):
                await forwarder.send(command)
                safe_commands.append((time.monotonic(), command))

        receiver.start()
        cm5_service = SafetyCommandService(
            receiver,
            ObservingForwarder(),
            tick_period_s=SETPOINT_PERIOD_S,
            obstacle_distance=distance_sensor.read,
        )
        cm5_service.start()
        cm5_task = asyncio.create_task(cm5_service.run(cm5_stop))
        sender = UdpCommandSender("127.0.0.1", receiver.port)

        vision = LatestVisionPipeline(
            PersonVisionPipeline(YoloPersonDetector(model_path="yolov8n.pt"))
        )
        control = CompanionRuntime(
            vision,
            ReactiveController(
                VisualFollower(
                    FollowConfig(
                        frame_width_px=video_config.width,
                        desired_target_height_px=video_config.height * 0.75,
                    )
                )
            )
        )
        flight_started_at = time.monotonic()

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
            sender,
            read_frame,
            intent_provider=current_intent,
            obstacle_provider=current_obstacle,
            tick_period_s=SETPOINT_PERIOD_S,
        )
        mac_task = asyncio.create_task(mac_service.run(mac_stop))
        deadline = time.monotonic() + 5.0
        while not safe_commands and time.monotonic() < deadline:
            await asyncio.sleep(SETPOINT_PERIOD_S)
        if not safe_commands:
            raise RuntimeError("CM5 did not forward a priming setpoint")
        print("CM5 priming setpoint=verified.")
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
        vision.close()
        cm5_stop.set()
        await cm5_task
        if not safe_commands or safe_commands[-1][1] != VelocityCommand():
            raise RuntimeError("full stack did not observe CM5 shutdown zero")
        if frames_received == 0:
            raise RuntimeError("full stack did not receive a decoded RTP frame")
        print("CM5 shutdown zero=verified.")
        print(f"RTP frames received: {frames_received}.")

        def forward_count(start_s, end_s):
            return sum(
                command.north_m_s > 0.0
                for timestamp_s, command in safe_commands
                if start_s <= timestamp_s - flight_started_at < end_s
            )

        print(
            "Following commands by cycle: "
            f"{forward_count(0.0, FOLLOW_END_S)}, "
            f"{forward_count(SECOND_FOLLOW_START_S, SECOND_FOLLOW_END_S)}, "
            f"{forward_count(THIRD_FOLLOW_START_S, THIRD_FOLLOW_END_S)}."
        )
        if min_north_velocity >= -0.05:
            raise RuntimeError(
                f"full stack did not observe obstacle backoff: {min_north_velocity:.2f}m/s"
            )
        if max_north_velocity <= 0.02:
            raise RuntimeError(
                f"full stack did not observe visual following: {max_north_velocity:.2f}m/s"
            )
        if max_east_velocity <= 0.02 and min_east_velocity >= -0.02:
            raise RuntimeError(
                "full stack did not observe lateral visual tracking: "
                f"{min_east_velocity:.2f}..{max_east_velocity:.2f}m/s"
            )

        def observed(start_s, end_s, predicate):
            return any(
                start_s <= timestamp_s - flight_started_at < end_s
                and predicate(command)
                for timestamp_s, command in safe_commands
            )

        objectives = (
            (5.0, 7.0, lambda command: command == VelocityCommand(), "hover intent"),
            (
                SECOND_FOLLOW_START_S,
                SECOND_FOLLOW_END_S,
                lambda command: command.north_m_s > 0.0,
                "following after hover",
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
                lambda command: command.north_m_s > 0.0,
                "following recovery after invalid sensor",
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
                lambda command: command.north_m_s > 0.0,
                "following recovery after target loss",
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
                lambda command: command.north_m_s > 0.0,
                "following after second hover",
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
        if telemetry_task is not None:
            telemetry_task.cancel()
            await asyncio.gather(telemetry_task, return_exceptions=True)
        if offboard_task is not None and not offboard_task.done():
            offboard_task.cancel()
            await asyncio.gather(offboard_task, return_exceptions=True)
        if vision is not None:
            vision.close()
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
        close_mavsdk(drone)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m sim.offboard_full IMAGE_PATH")
    asyncio.run(run(sys.argv[1]))
