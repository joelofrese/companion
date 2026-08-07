"""Shared PX4 flight steps for the simulations."""

import asyncio
import math
import time

from mavsdk.telemetry import FlightMode
from mavsdk.telemetry import LandedState


TAKEOFF_ALTITUDE = 2.0
TAKEOFF_YAW_MODE = 5
TAKEOFF_HEADING_LIMIT_DEG = 15.0
PREPARE_TIMEOUT_S = 30.0
FLIGHT_ACTION_TIMEOUT_S = 120.0
VELOCITY_TELEMETRY_RATE_HZ = 10.0
ATTITUDE_TELEMETRY_RATE_HZ = 5.0


class RecordingForwarder:
    """Forward commands and keep them for later checks."""

    def __init__(self, forwarder):
        self.forwarder = forwarder
        self.commands = []

    async def send(self, command):
        await self.forwarder.send(command)
        self.commands.append((time.monotonic(), command))


async def _wait_for_connection(drone):
    async for state in drone.core.connection_state():
        if state.is_connected:
            return


async def _connect_and_wait(drone):
    await drone.connect()
    await _wait_for_connection(drone)


async def _wait_until_ready(drone):
    async for health in drone.telemetry.health():
        if (
            health.is_local_position_ok
            and health.is_magnetometer_calibration_ok
        ):
            return


async def _wait_until_armed(drone):
    async for armed in drone.telemetry.armed():
        if armed:
            return


async def _wait_until_in_air(drone):
    async for state in drone.telemetry.landed_state():
        if state == LandedState.IN_AIR:
            return


async def _wait_until_on_ground(drone):
    async for state in drone.telemetry.landed_state():
        if state == LandedState.ON_GROUND:
            return


async def _wait_until_disarmed(drone):
    async for armed in drone.telemetry.armed():
        if not armed:
            return


async def _read_heading(drone):
    async for attitude in drone.telemetry.attitude_euler():
        if math.isfinite(attitude.yaw_deg):
            return attitude.yaw_deg


def _heading_change_deg(start_deg: float, end_deg: float) -> float:
    return abs((end_deg - start_deg + 180.0) % 360.0 - 180.0)


async def prepare(drone):
    """Connect, wait for arming health, arm, and reach the air.

    Return the heading that every simulated velocity setpoint must hold.
    """

    print("Waiting for drone connection...")
    try:
        await asyncio.wait_for(_connect_and_wait(drone), PREPARE_TIMEOUT_S)
    except asyncio.TimeoutError as error:
        raise RuntimeError(
            f"drone did not connect within {PREPARE_TIMEOUT_S:.0f}s"
        ) from error
    print("Connected.")

    print("Waiting for vehicle to be ready to arm...")
    try:
        await asyncio.wait_for(_wait_until_ready(drone), PREPARE_TIMEOUT_S)
    except asyncio.TimeoutError as error:
        raise RuntimeError(
            f"vehicle did not become ready to arm within {PREPARE_TIMEOUT_S:.0f}s"
        ) from error
    print("Ready.")
    await drone.telemetry.set_rate_velocity_ned(VELOCITY_TELEMETRY_RATE_HZ)
    await drone.telemetry.set_rate_attitude_euler(ATTITUDE_TELEMETRY_RATE_HZ)
    # Keep the vehicle's current heading during PX4's automatic takeoff.
    await drone.param.set_param_int("MPC_YAW_MODE", TAKEOFF_YAW_MODE)
    try:
        await asyncio.wait_for(_wait_until_ready(drone), PREPARE_TIMEOUT_S)
    except asyncio.TimeoutError as error:
        raise RuntimeError(
            "vehicle did not become ready after takeoff heading setup "
            f"within {PREPARE_TIMEOUT_S:.0f}s"
        ) from error
    try:
        initial_heading_deg = await asyncio.wait_for(
            _read_heading(drone),
            PREPARE_TIMEOUT_S,
        )
    except asyncio.TimeoutError as error:
        raise RuntimeError(
            "vehicle did not provide heading before takeoff within "
            f"{PREPARE_TIMEOUT_S:.0f}s"
        ) from error

    print("Arming...")
    try:
        await drone.action.arm()
        await asyncio.wait_for(_wait_until_armed(drone), FLIGHT_ACTION_TIMEOUT_S)
        print(f"Taking off to {TAKEOFF_ALTITUDE}m...")
        await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE)
        await drone.action.takeoff()
        try:
            await asyncio.wait_for(
                _wait_until_in_air(drone),
                FLIGHT_ACTION_TIMEOUT_S,
            )
        except asyncio.TimeoutError as error:
            raise RuntimeError(
                f"vehicle did not take off within {FLIGHT_ACTION_TIMEOUT_S:.0f}s"
            ) from error
        try:
            takeoff_heading_deg = await asyncio.wait_for(
                _read_heading(drone),
                PREPARE_TIMEOUT_S,
            )
        except asyncio.TimeoutError as error:
            raise RuntimeError(
                "vehicle did not provide heading after takeoff within "
                f"{PREPARE_TIMEOUT_S:.0f}s"
            ) from error
        heading_change_deg = _heading_change_deg(
            initial_heading_deg,
            takeoff_heading_deg,
        )
        if heading_change_deg > TAKEOFF_HEADING_LIMIT_DEG:
            raise RuntimeError(
                "PX4 rotated during automatic takeoff: "
                f"{heading_change_deg:.1f} degrees"
            )
        print(
            "PX4 takeoff heading hold=verified: "
            f"change {heading_change_deg:.1f} degrees."
        )
        return initial_heading_deg
    except Exception:
        try:
            await asyncio.wait_for(
                drone.action.land(),
                FLIGHT_ACTION_TIMEOUT_S,
            )
        except Exception:
            pass
        raise


async def land(drone):
    """Land and wait for PX4 to report the vehicle on the ground."""

    print("Landing...")
    deadline = time.monotonic() + FLIGHT_ACTION_TIMEOUT_S
    while True:
        try:
            await drone.action.land()
            remaining = deadline - time.monotonic()
            await asyncio.wait_for(
                _wait_until_on_ground(drone),
                max(0.1, min(5.0, remaining)),
            )
            break
        except asyncio.TimeoutError as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"vehicle did not land within {FLIGHT_ACTION_TIMEOUT_S:.0f}s"
                ) from error
    print("Landed.")
    try:
        await asyncio.wait_for(
            _wait_until_disarmed(drone),
            FLIGHT_ACTION_TIMEOUT_S,
        )
    except asyncio.TimeoutError as error:
        raise RuntimeError(
            f"vehicle did not disarm within {FLIGHT_ACTION_TIMEOUT_S:.0f}s"
        ) from error
    print("Disarmed.")


async def wait_for_offboard(drone):
    """Wait for PX4 telemetry to confirm offboard mode is active."""

    async for mode in drone.telemetry.flight_mode():
        if mode is FlightMode.OFFBOARD:
            return


def close_mavsdk(drone):
    """Stop the MAVSDK helper started by the Python binding."""

    drone._stop_mavsdk_server()
