"""
Minimal hover test — connects to PX4 SITL, arms, takes off, hovers, lands.
Run after launching PX4 SITL: make px4_sitl gz_x500
"""

import asyncio
from mavsdk import System

HOVER_DURATION = 15     # seconds

from sim.flight import close_mavsdk, land, prepare


async def run():
    drone = System()
    try:
        await prepare(drone)
        print(f"Hovering for {HOVER_DURATION}s...")
        await asyncio.sleep(HOVER_DURATION)
        await land(drone)
    finally:
        close_mavsdk(drone)


if __name__ == "__main__":
    asyncio.run(run())
