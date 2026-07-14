"""
Minimal hover test — connects to PX4 SITL, arms, takes off, hovers, lands.
Run after launching PX4 SITL: make px4_sitl gz_x500
"""

import asyncio
from mavsdk import System
from mavsdk.telemetry import LandedState


TAKEOFF_ALTITUDE = 2.0  # meters
HOVER_DURATION = 15     # seconds


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

    print(f"Taking off to {TAKEOFF_ALTITUDE}m...")
    await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE)
    await drone.action.takeoff()

    print(f"Hovering for {HOVER_DURATION}s...")
    await asyncio.sleep(HOVER_DURATION)

    print("Landing...")
    await drone.action.land()
    async for state in drone.telemetry.landed_state():
        if state == LandedState.ON_GROUND:
            print("Landed.")
            break


if __name__ == "__main__":
    asyncio.run(run())
