"""Turn VLM movement suggestions into slow Mac commands."""

import math
from numbers import Real
from typing import Optional

from control.safety_limits import OBSTACLE_STOP_M
from control.velocity import VelocityCommand


MOVEMENT_COMMANDS = {
    "forward": VelocityCommand(north_m_s=0.25),
    "left": VelocityCommand(east_m_s=-0.2),
    "right": VelocityCommand(east_m_s=0.2),
    "up": VelocityCommand(down_m_s=-0.15),
    "down": VelocityCommand(down_m_s=0.15),
    "stop": VelocityCommand(),
    "hover": VelocityCommand(),
}


def movement_command(
    movement: str,
    obstacle_distance_m: Optional[float] = None,
) -> VelocityCommand:
    """Return one bounded command or zero for unsafe input."""

    if obstacle_distance_m is not None:
        if (
            isinstance(obstacle_distance_m, bool)
            or not isinstance(obstacle_distance_m, Real)
            or not math.isfinite(obstacle_distance_m)
            or obstacle_distance_m < 0.0
        ):
            return VelocityCommand()
        if obstacle_distance_m < OBSTACLE_STOP_M:
            return VelocityCommand()
    if not isinstance(movement, str):
        return VelocityCommand()
    return MOVEMENT_COMMANDS.get(movement.strip().lower(), VelocityCommand())
