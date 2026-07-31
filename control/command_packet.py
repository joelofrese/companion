"""Encode the small Mac-to-CM5 UDP contract."""

import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional

from control.velocity import VelocityCommand


PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 512


@dataclass(frozen=True)
class CommandPacket:
    """One numbered velocity command."""

    sequence: int
    command: VelocityCommand

    def encode(self) -> bytes:
        values = (
            self.command.north_m_s,
            self.command.east_m_s,
            self.command.down_m_s,
            self.command.yaw_deg,
        )
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("command velocity must be finite")
        payload = {
            "version": PROTOCOL_VERSION,
            "sequence": self.sequence,
            "velocity": {
                "north_m_s": self.command.north_m_s,
                "east_m_s": self.command.east_m_s,
                "down_m_s": self.command.down_m_s,
                "yaw_deg": self.command.yaw_deg,
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
            values = tuple(velocity[name] for name in ("north_m_s", "east_m_s", "down_m_s", "yaw_deg"))
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
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("invalid command velocity")
        return cls(sequence, VelocityCommand(*values))


@dataclass(frozen=True)
class TelemetryPacket:
    """Return the newest CM5 distance reading to the Mac."""

    sequence: int
    obstacle_distance_m: Optional[float]

    def encode(self) -> bytes:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.obstacle_distance_m is not None and (
            isinstance(self.obstacle_distance_m, bool)
            or not isinstance(self.obstacle_distance_m, Real)
            or not math.isfinite(self.obstacle_distance_m)
            or self.obstacle_distance_m < 0.0
        ):
            raise ValueError("obstacle distance must be finite or none")
        payload = {
            "type": "telemetry",
            "version": PROTOCOL_VERSION,
            "sequence": self.sequence,
            "telemetry": {"obstacle_distance_m": self.obstacle_distance_m},
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
            distance = data["telemetry"]["obstacle_distance_m"]
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
        if distance is not None and (
            isinstance(distance, bool)
            or not isinstance(distance, Real)
            or not math.isfinite(distance)
            or distance < 0.0
        ):
            raise ValueError("invalid obstacle distance")
        return cls(sequence, distance)
