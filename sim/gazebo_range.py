"""Decode a forward range sensor from Gazebo."""

import math

from sim.gazebo_topic import GazeboTopicReader
from sim.offboard_control import DistanceMessage


class GazeboRangefinder(GazeboTopicReader):
    """Keep the newest forward Gazebo lidar reading."""

    def __init__(self, topic: str):
        super().__init__(topic, _decode_range, "lidar")


def _decode_range(message):
    minimum = _finite_number(message["rangeMin"])
    maximum = _finite_number(message["rangeMax"])
    values = [
        value
        for value in (_number_or_nan(item) for item in message["ranges"])
        if math.isfinite(value)
    ]
    return DistanceMessage(
        min(values) if values else math.nan,
        minimum,
        maximum,
    )


def _finite_number(value) -> float:
    if isinstance(value, bool):
        raise ValueError("range limit must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("range limit must be finite")
    return number


def _number_or_nan(value) -> float:
    if isinstance(value, bool):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan
