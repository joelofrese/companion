"""Stop motion when a Mac command arrives too late."""

import math
from numbers import Real

from control.velocity import VelocityCommand


class SetpointWatchdog:
    """Send zero once when a command misses its deadline."""

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

    def emit(self, timestamp_s: float, desired: VelocityCommand) -> VelocityCommand:
        """Return the command allowed at this time."""

        if self._last_sent_at_s is not None:
            interval_s = timestamp_s - self._last_sent_at_s
            if interval_s <= 0.0:
                raise ValueError("watchdog timestamps must increase")
            if interval_s > self.max_interval_s:
                self._last_sent_at_s = timestamp_s
                return VelocityCommand()
        self._last_sent_at_s = timestamp_s
        return desired
