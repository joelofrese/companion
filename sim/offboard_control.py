"""Small, deterministic reactive velocity profile for SITL and hardware bring-up."""

from control.velocity import VelocityCommand


def demo_velocity(elapsed_s: float) -> VelocityCommand:
    """Move forward slowly for four seconds, then hold a zero-velocity hover."""

    if 0.0 <= elapsed_s < 4.0:
        return VelocityCommand(north_m_s=0.5)
    return VelocityCommand()
