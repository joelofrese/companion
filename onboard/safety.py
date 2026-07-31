"""Apply command and sensor safety on the CM5 before forwarding."""

import math
import threading
import time
from numbers import Real
from typing import Optional

from control.command_packet import CommandPacket
from control.safety_limits import BACKOFF_SPEED_M_S, OBSTACLE_STOP_M
from control.velocity import VelocityCommand


MAX_HORIZONTAL_SPEED_M_S = 0.5
MAX_VERTICAL_SPEED_M_S = 0.3
MAX_YAW_DEG = 180.0
DISTANCE_TIMEOUT_S = 0.15


def _finite_real(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(value)
    )


class LatestDistanceSensor:
    """Keep the newest distance reading; missing data is unsafe."""

    def __init__(self, clock=time.monotonic, timeout_s: float = DISTANCE_TIMEOUT_S):
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, Real)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
        ):
            raise ValueError("distance timeout must be positive")
        self._distance_m = math.nan
        self._clock = clock
        self._timeout_s = timeout_s
        self._updated_at_s = None
        self._lock = threading.Lock()

    def update(self, message):
        distance_m = getattr(message, "current_distance", math.nan)
        minimum_m = getattr(message, "min_distance", None)
        maximum_m = getattr(message, "max_distance", None)
        valid = (
            _finite_real(distance_m)
            and distance_m >= 0.0
            and (minimum_m is None or (_finite_real(minimum_m) and minimum_m >= 0.0))
            and (maximum_m is None or (_finite_real(maximum_m) and maximum_m >= 0.0))
            and (minimum_m is None or maximum_m is None or minimum_m <= maximum_m)
            and (minimum_m is None or distance_m >= minimum_m)
            and (maximum_m is None or distance_m <= maximum_m)
        )
        if not valid:
            distance_m = math.nan
        with self._lock:
            self._distance_m = distance_m
            self._updated_at_s = self._clock()

    def read(self):
        now = self._clock()
        with self._lock:
            if (
                self._updated_at_s is None
                or not _finite_real(now)
                or now < self._updated_at_s
                or now - self._updated_at_s > self._timeout_s
            ):
                return math.nan
            return self._distance_m


class OnboardSafetyEnvelope:
    """Expire old commands and stop for a forward obstacle."""

    def __init__(self, command_timeout_s: float = 0.15):
        if (
            isinstance(command_timeout_s, bool)
            or not isinstance(command_timeout_s, Real)
            or not math.isfinite(command_timeout_s)
            or command_timeout_s <= 0.0
        ):
            raise ValueError("command timeout must be positive")
        self.command_timeout_s = command_timeout_s
        self._command: Optional[VelocityCommand] = None
        self._received_at_s: Optional[float] = None
        self._last_tick_s: Optional[float] = None
        self._last_sequence: Optional[int] = None

    def receive(self, timestamp_s: float, command: VelocityCommand):
        """Record one command using the CM5 receive timestamp."""

        if (
            isinstance(timestamp_s, bool)
            or not isinstance(timestamp_s, Real)
            or not math.isfinite(timestamp_s)
        ):
            raise ValueError("received command timestamp must be finite")
        if self._received_at_s is not None and timestamp_s <= self._received_at_s:
            raise ValueError("received command timestamps must increase")
        self._command = command
        self._received_at_s = timestamp_s

    def receive_packet(self, timestamp_s: float, payload: bytes) -> bool:
        """Accept a newer wire packet and ignore duplicates or reordered packets."""

        packet = CommandPacket.decode(payload)
        if self._last_sequence is not None and packet.sequence <= self._last_sequence:
            return False
        self.receive(timestamp_s, packet.command)
        self._last_sequence = packet.sequence
        return True

    def tick(
        self,
        timestamp_s: float,
        obstacle_distance_m: Optional[float] = None,
    ) -> VelocityCommand:
        """Return the command safe to forward to PX4 at this local tick."""

        if (
            isinstance(timestamp_s, bool)
            or not isinstance(timestamp_s, Real)
            or not math.isfinite(timestamp_s)
        ):
            return VelocityCommand()
        if self._last_tick_s is not None and timestamp_s <= self._last_tick_s:
            raise ValueError("safety tick timestamps must increase")
        self._last_tick_s = timestamp_s

        if self._command is None or self._received_at_s is None:
            return VelocityCommand()
        if timestamp_s - self._received_at_s > self.command_timeout_s:
            return VelocityCommand()
        if obstacle_distance_m is not None and (
            not _finite_real(obstacle_distance_m) or obstacle_distance_m < 0.0
        ):
            return VelocityCommand()
        if obstacle_distance_m is not None and obstacle_distance_m < OBSTACLE_STOP_M:
            return VelocityCommand(north_m_s=-BACKOFF_SPEED_M_S)
        if not self._command_is_safe(self._command):
            return VelocityCommand()
        return self._command

    @staticmethod
    def _command_is_safe(command: VelocityCommand) -> bool:
        values = (
            command.north_m_s,
            command.east_m_s,
            command.down_m_s,
            command.yaw_deg,
        )
        if not all(_finite_real(value) for value in values):
            return False
        return (
            abs(command.north_m_s) <= MAX_HORIZONTAL_SPEED_M_S
            and abs(command.east_m_s) <= MAX_HORIZONTAL_SPEED_M_S
            and abs(command.down_m_s) <= MAX_VERTICAL_SPEED_M_S
            and abs(command.yaw_deg) <= MAX_YAW_DEG
        )
