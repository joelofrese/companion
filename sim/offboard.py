"""Continuous MAVSDK offboard velocity smoke test for PX4 SITL."""

import asyncio
import time
from typing import Awaitable, Callable, Optional

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
from mavsdk.telemetry import LandedState

from control.loop import CompanionControlLoop
from control.runtime import CompanionRuntime
from control.step import CompanionControlStep
from sim.offboard_control import DemoVision, demo_obstacle_distance_m, demo_state


SETPOINT_PERIOD_S = 0.05
TAKEOFF_ALTITUDE = 2.0
PROFILE_DURATION_S = 8.0


def _mavsdk_velocity(command):
    return VelocityNedYaw(
        command.north_m_s,
        command.east_m_s,
        command.down_m_s,
        command.yaw_deg,
    )


async def _wait_until_in_air(drone):
    async for state in drone.telemetry.landed_state():
        if state == LandedState.IN_AIR:
            return


FrameReader = Callable[[], Awaitable[Optional[tuple[float, object]]]]


async def _read_frame(frame_reader: Optional[FrameReader]):
    if frame_reader is None:
        return time.monotonic(), None
    result = await frame_reader()
    if result is None:
        raise RuntimeError("video stream ended before the offboard scenario completed")
    return result


async def run(
    vision=None,
    runtime: Optional[CompanionRuntime] = None,
    frame_reader: Optional[FrameReader] = None,
):
    print("Waiting for drone connection...")
    drone = System()
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
    await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE)
    await drone.action.takeoff()
    await _wait_until_in_air(drone)

    # PX4 requires a setpoint before offboard.start and a continuous stream after it.
    control_step = CompanionControlStep(vision or DemoVision(), runtime=runtime)
    now, frame = await _read_frame(frame_reader)
    command = control_step.process(
        frame=frame,
        timestamp_s=now,
        intent=demo_state(0.0),
        obstacle_distance_m=demo_obstacle_distance_m(0.0),
    )
    await drone.offboard.set_velocity_ned(_mavsdk_velocity(command))
    await drone.offboard.start()
    print("Offboard started.")
    control_loop = CompanionControlLoop(control_step)

    max_north_velocity = 0.0
    min_north_velocity = 0.0
    obstacle_active = False

    async def observe_velocity():
        nonlocal max_north_velocity, min_north_velocity
        async for velocity in drone.telemetry.velocity_ned():
            max_north_velocity = max(max_north_velocity, velocity.north_m_s)
            min_north_velocity = min(min_north_velocity, velocity.north_m_s)

    telemetry_task = asyncio.create_task(observe_velocity())
    started_at = time.monotonic()
    try:
        while (elapsed := time.monotonic() - started_at) < PROFILE_DURATION_S:
            now, frame = await _read_frame(frame_reader)
            obstacle_distance_m = demo_obstacle_distance_m(elapsed)
            if obstacle_distance_m < 0.6 and not obstacle_active:
                print("Obstacle detected; backing off.")
                obstacle_active = True
            elif obstacle_distance_m >= 0.6:
                obstacle_active = False
            command = control_loop.tick(
                frame=frame,
                timestamp_s=now,
                intent=demo_state(elapsed),
                obstacle_distance_m=obstacle_distance_m,
            )
            if control_loop.watchdog.tripped:
                print("Setpoint watchdog tripped; sending zero velocity and landing.")
                await drone.offboard.set_velocity_ned(_mavsdk_velocity(command))
                break
            await drone.offboard.set_velocity_ned(_mavsdk_velocity(command))
            await asyncio.sleep(SETPOINT_PERIOD_S)
    finally:
        now, frame = await _read_frame(frame_reader)
        command = control_loop.tick(
            frame=frame,
            timestamp_s=now,
            intent=demo_state(PROFILE_DURATION_S),
            obstacle_distance_m=demo_obstacle_distance_m(PROFILE_DURATION_S),
        )
        await drone.offboard.set_velocity_ned(_mavsdk_velocity(command))
        telemetry_task.cancel()
        await asyncio.gather(telemetry_task, return_exceptions=True)
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass

    if min_north_velocity >= -0.05:
        raise RuntimeError(f"SITL did not observe obstacle backoff velocity: {min_north_velocity:.2f}m/s")
    print(f"Max observed north velocity: {max_north_velocity:.2f}m/s")
    print(f"Min observed north velocity: {min_north_velocity:.2f}m/s")
    print("Landing...")
    await drone.action.land()
    async for state in drone.telemetry.landed_state():
        if state == LandedState.ON_GROUND:
            print("Landed.")
            break


if __name__ == "__main__":
    asyncio.run(run())
