"""Verify RTP vision, Mac UDP control, CM5 safety, and PX4 SITL together."""

import asyncio
import subprocess
import sys
import time

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw
from mavsdk.telemetry import LandedState

from control.following import FollowConfig, VisualFollower
from control.runtime import CompanionRuntime
from control.state_machine import ReactiveController
from control.udp_control import UdpControlService
from control.udp_sender import UdpCommandSender
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from onboard.velocity_forwarder import MavsdkVelocityForwarder
from sim.offboard import PROFILE_DURATION_S, SETPOINT_PERIOD_S, TAKEOFF_ALTITUDE, _wait_until_in_air
from sim.offboard_control import demo_obstacle_distance_m, demo_state
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


def _mavsdk_velocity(command):
    return VelocityNedYaw(
        command.north_m_s,
        command.east_m_s,
        command.down_m_s,
        command.yaw_deg,
    )


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
    cm5_stop = asyncio.Event()
    mac_stop = asyncio.Event()
    obstacle_distance_m = 2.0
    flight_started_at = None
    armed = False
    offboard_started = False
    landed = False
    max_north_velocity = 0.0
    min_north_velocity = 0.0

    def current_obstacle(timestamp_s):
        nonlocal obstacle_distance_m
        elapsed_s = max(0.0, timestamp_s - flight_started_at)
        obstacle_distance_m = demo_obstacle_distance_m(elapsed_s)
        return obstacle_distance_m

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

        print("Waiting for drone connection...")
        await drone.connect()
        async for state in drone.core.connection_state():
            if state.is_connected:
                print("Connected.")
                break

        print("Waiting for vehicle to be ready to arm...")
        async for health in drone.telemetry.health():
            if (health.is_global_position_ok
                    and health.is_home_position_ok
                    and health.is_magnetometer_calibration_ok):
                print("Ready.")
                break

        print("Arming...")
        await drone.action.arm()
        armed = True
        await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE)
        await drone.action.takeoff()
        await _wait_until_in_air(drone)

        forwarder = MavsdkVelocityForwarder(drone)
        receiver.start()
        cm5_service = SafetyCommandService(
            receiver,
            forwarder,
            tick_period_s=SETPOINT_PERIOD_S,
            obstacle_distance=lambda: obstacle_distance_m,
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
                        desired_target_height_px=video_config.height / 4.0,
                    )
                )
            )
        )
        flight_started_at = time.monotonic()
        mac_service = UdpControlService(
            control,
            sender,
            frame_reader.read,
            intent_provider=current_intent,
            obstacle_provider=current_obstacle,
            tick_period_s=SETPOINT_PERIOD_S,
        )
        mac_task = asyncio.create_task(mac_service.run(mac_stop))
        await asyncio.sleep(0.2)
        await drone.offboard.start()
        offboard_started = True
        print("Offboard started through full Mac/CM5 stack.")

        async def observe_velocity():
            nonlocal max_north_velocity, min_north_velocity
            async for velocity in drone.telemetry.velocity_ned():
                max_north_velocity = max(max_north_velocity, velocity.north_m_s)
                min_north_velocity = min(min_north_velocity, velocity.north_m_s)

        telemetry_task = asyncio.create_task(observe_velocity())
        await asyncio.sleep(PROFILE_DURATION_S)
        mac_stop.set()
        await mac_task
        vision.close()
        cm5_stop.set()
        await cm5_task
        if min_north_velocity >= -0.05:
            raise RuntimeError(
                f"full stack did not observe obstacle backoff: {min_north_velocity:.2f}m/s"
            )
        print(f"Max observed north velocity: {max_north_velocity:.2f}m/s")
        print(f"Min observed north velocity: {min_north_velocity:.2f}m/s")
        await drone.offboard.stop()
        offboard_started = False
        print("Landing...")
        await drone.action.land()
        async for state in drone.telemetry.landed_state():
            if state == LandedState.ON_GROUND:
                landed = True
                print("Landed.")
                break
    finally:
        await _stop_task(mac_task, mac_stop)
        await _stop_task(cm5_task, cm5_stop)
        if telemetry_task is not None:
            telemetry_task.cancel()
            await asyncio.gather(telemetry_task, return_exceptions=True)
        if vision is not None:
            vision.close()
        if offboard_started:
            try:
                await drone.offboard.stop()
            except Exception:
                pass
        if armed and not landed:
            try:
                await drone.action.land()
                async for state in drone.telemetry.landed_state():
                    if state == LandedState.ON_GROUND:
                        break
            except Exception:
                pass
        receiver.close()
        video_receiver.close()
        if camera_process is not None:
            close_subprocess(camera_process)
        if sender is not None:
            sender.close()
        drone._stop_mavsdk_server()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m sim.offboard_full IMAGE_PATH")
    asyncio.run(run(sys.argv[1]))
