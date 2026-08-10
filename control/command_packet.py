"""Encode the small brain-to-CM5 UDP contract."""

import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional

from control.velocity import VelocityCommand


PROTOCOL_VERSION = 4
MAX_PACKET_BYTES = 512


@dataclass(frozen=True)
class CommandPacket:
    """One numbered movement command."""

    sequence: int
    command: VelocityCommand

    def encode(self) -> bytes:
        values = (
            self.command.forward_m_s,
            self.command.right_m_s,
            self.command.down_m_s,
            self.command.yaw_rate_deg_s,
        )
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be non-negative")
        if any(not _finite(value) for value in values):
            raise ValueError("command velocity must be finite")
        payload = {
            "version": PROTOCOL_VERSION,
            "sequence": self.sequence,
            "velocity": {
                "forward_m_s": self.command.forward_m_s,
                "right_m_s": self.command.right_m_s,
                "down_m_s": self.command.down_m_s,
                "yaw_rate_deg_s": self.command.yaw_rate_deg_s,
            },
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(encoded) > MAX_PACKET_BYTES:
            raise ValueError("command packet is too large")
        return encoded

    @classmethod
    def decode(cls, payload: bytes) -> "CommandPacket":
        if len(payload) > MAX_PACKET_BYTES:
            raise ValueError("command packet is too large")
        try:
            data = json.loads(payload.decode("utf-8"))
            version = data["version"]
            sequence = data["sequence"]
            velocity = data["velocity"]
            values = tuple(
                velocity[name]
                for name in (
                    "forward_m_s",
                    "right_m_s",
                    "down_m_s",
                    "yaw_rate_deg_s",
                )
            )
        except (
            AttributeError,
            KeyError,
            IndexError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError("invalid command packet") from error

        if (
            version != PROTOCOL_VERSION
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise ValueError("invalid command packet header")
        if any(not _finite(value) for value in values):
            raise ValueError("invalid command velocity")
        return cls(sequence, VelocityCommand(*values))


@dataclass(frozen=True)
class TelemetryPacket:
    """Return the newest CM5 sensor and body velocity readings to the brain."""

    sequence: int
    obstacle_distance_m: Optional[float]
    forward_velocity_m_s: Optional[float] = None
    right_velocity_m_s: Optional[float] = None
    down_velocity_m_s: Optional[float] = None

    def encode(self) -> bytes:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be non-negative")
        if self.obstacle_distance_m is not None and not _distance(self.obstacle_distance_m):
            raise ValueError("obstacle distance must be finite or none")
        velocities = (
            self.forward_velocity_m_s,
            self.right_velocity_m_s,
            self.down_velocity_m_s,
        )
        if any(value is not None and not _finite(value) for value in velocities):
            raise ValueError("velocity telemetry must be finite or none")
        payload = {
            "type": "telemetry",
            "version": PROTOCOL_VERSION,
            "sequence": self.sequence,
            "telemetry": {
                "obstacle_distance_m": self.obstacle_distance_m,
                "forward_velocity_m_s": self.forward_velocity_m_s,
                "right_velocity_m_s": self.right_velocity_m_s,
                "down_velocity_m_s": self.down_velocity_m_s,
            },
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(encoded) > MAX_PACKET_BYTES:
            raise ValueError("telemetry packet is too large")
        return encoded

    @classmethod
    def decode(cls, payload: bytes) -> "TelemetryPacket":
        if len(payload) > MAX_PACKET_BYTES:
            raise ValueError("telemetry packet is too large")
        try:
            data = json.loads(payload.decode("utf-8"))
            version = data["version"]
            packet_type = data["type"]
            sequence = data["sequence"]
            telemetry = data["telemetry"]
            distance = telemetry["obstacle_distance_m"]
            velocities = tuple(
                telemetry.get(name)
                for name in (
                    "forward_velocity_m_s",
                    "right_velocity_m_s",
                    "down_velocity_m_s",
                )
            )
        except (
            AttributeError,
            KeyError,
            IndexError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError("invalid telemetry packet") from error

        if (
            version != PROTOCOL_VERSION
            or packet_type != "telemetry"
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise ValueError("invalid telemetry packet header")
        if distance is not None and not _distance(distance):
            raise ValueError("invalid obstacle distance")
        if any(value is not None and not _finite(value) for value in velocities):
            raise ValueError("invalid velocity telemetry")
        return cls(sequence, distance, *velocities)


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(value)
    )


def _distance(value: object) -> bool:
    return _finite(value) and value >= 0.0
