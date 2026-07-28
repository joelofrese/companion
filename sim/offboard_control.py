"""Deterministic cognitive intent profile for SITL bring-up."""

from control.state_machine import State


def demo_state(elapsed_s: float) -> State:
    """Follow for four seconds, then ask the reactive layer to hover."""

    if 0.0 <= elapsed_s < 4.0:
        return State.FOLLOWING
    return State.HOVERING
