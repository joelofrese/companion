"""Continuous MAVSDK offboard velocity smoke test for PX4 SITL."""

import asyncio
import time

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
from mavsdk.telemetry import LandedState

from control.state_machine import ReactiveController
from sim.offboard_control import demo_state


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


async def run():
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
    controller = ReactiveController()
    controller.set_intent(demo_state(0.0))
    await drone.offboard.set_velocity_ned(_mavsdk_velocity(controller.command()))
    await drone.offboard.start()
    print("Offboard started.")

    max_north_velocity = 0.0

    async def observe_velocity():
        nonlocal max_north_velocity
        async for velocity in drone.telemetry.velocity_ned():
            max_north_velocity = max(max_north_velocity, abs(velocity.north_m_s))

    telemetry_task = asyncio.create_task(observe_velocity())
    started_at = time.monotonic()
    try:
        while (elapsed := time.monotonic() - started_at) < PROFILE_DURATION_S:
            controller.set_intent(demo_state(elapsed))
            await drone.offboard.set_velocity_ned(_mavsdk_velocity(controller.command()))
            await asyncio.sleep(SETPOINT_PERIOD_S)
    finally:
        controller.set_intent(demo_state(PROFILE_DURATION_S))
        await drone.offboard.set_velocity_ned(_mavsdk_velocity(controller.command()))
        telemetry_task.cancel()
        await asyncio.gather(telemetry_task, return_exceptions=True)
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass

    print(f"Max observed north velocity: {max_north_velocity:.2f}m/s")
    print("Landing...")
    await drone.action.land()
    async for state in drone.telemetry.landed_state():
        if state == LandedState.ON_GROUND:
            print("Landed.")
            break


if __name__ == "__main__":
    asyncio.run(run())
