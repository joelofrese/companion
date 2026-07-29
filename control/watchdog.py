"""Latched setpoint heartbeat watchdog for safety-critical offboard output."""

import math
from numbers import Real

from control.velocity import VelocityCommand


class SetpointWatchdog:
    """Replace commands with zero velocity after a missed setpoint deadline."""

    def __init__(self, max_interval_s: float = 0.15):
        if (
            isinstance(max_interval_s, bool)
            or not isinstance(max_interval_s, Real)
            or not math.isfinite(max_interval_s)
            or max_interval_s <= 0.0
        ):
            raise ValueError("watchdog interval must be positive")
        self.max_interval_s = max_interval_s
        self._last_sent_at_s = None
        self.tripped = False

    def emit(self, timestamp_s: float, desired: VelocityCommand) -> VelocityCommand:
        """Return the command allowed at this tick and latch any missed deadline."""

        if self._last_sent_at_s is not None:
            interval_s = timestamp_s - self._last_sent_at_s
            if interval_s <= 0.0:
                raise ValueError("watchdog timestamps must increase")
            if interval_s > self.max_interval_s:
                self.tripped = True
        self._last_sent_at_s = timestamp_s
        return VelocityCommand() if self.tripped else desired
