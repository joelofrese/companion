"""Track a person in image coordinates."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional


@dataclass(frozen=True)
class Detection:
    """One person found in an image."""

    x_px: float
    y_px: float
    timestamp_s: float
    confidence: float = 1.0
    width_px: float = 0.0
    height_px: float = 0.0


@dataclass(frozen=True)
class TrackEstimate:
    """The current and predicted position of a person."""

    x_px: float
    y_px: float
    vx_px_s: float
    vy_px_s: float
    predicted_x_px: float
    predicted_y_px: float
    age_s: float = 0.0
    target_width_px: float = 0.0
    target_height_px: float = 0.0


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


class _AxisFilter:
    """Filter one image axis."""

    def __init__(self, process_noise: float, measurement_noise: float):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.position = 0.0
        self.velocity = 0.0
        self.position_variance = 1.0
        self.velocity_variance = 1.0
        self.covariance = 0.0

    def initialize(self, position: float):
        self.position = position
        self.velocity = 0.0

    def predict(self, dt: float):
        self.position += self.velocity * dt
        position_variance = (
            self.position_variance
            + 2.0 * dt * self.covariance
            + dt * dt * self.velocity_variance
        )
        covariance = self.covariance + dt * self.velocity_variance
        acceleration_variance = self.process_noise
        self.position_variance = position_variance + acceleration_variance * dt**4 / 4.0
        self.covariance = covariance + acceleration_variance * dt**3 / 2.0
        self.velocity_variance += acceleration_variance * dt * dt

    def update(self, measurement: float):
        innovation_variance = self.position_variance + self.measurement_noise
        position_gain = self.position_variance / innovation_variance
        velocity_gain = self.covariance / innovation_variance
        innovation = measurement - self.position
        covariance_prior = self.covariance
        self.position += position_gain * innovation
        self.velocity += velocity_gain * innovation
        self.position_variance *= 1.0 - position_gain
        self.covariance = covariance_prior * (1.0 - position_gain)
        self.velocity_variance -= velocity_gain * covariance_prior


class PersonTracker:
    """Track a person and predict its short-term position."""

    def __init__(self, prediction_horizon_s: float = 0.3, max_prediction_age_s: float = 0.5):
        if not _finite(prediction_horizon_s) or prediction_horizon_s < 0.0:
            raise ValueError("prediction horizon must be finite and non-negative")
        if not _finite(max_prediction_age_s) or max_prediction_age_s < 0.0:
            raise ValueError("maximum prediction age must be finite and non-negative")
        self.prediction_horizon_s = prediction_horizon_s
        self.max_prediction_age_s = max_prediction_age_s
        self._x = _AxisFilter(process_noise=1.0, measurement_noise=4.0)
        self._y = _AxisFilter(process_noise=1.0, measurement_noise=4.0)
        self._state_timestamp_s: Optional[float] = None
        self._last_measurement_timestamp_s: Optional[float] = None
        self._target_width_px = 0.0
        self._target_height_px = 0.0

    def update(self, detection: Detection) -> TrackEstimate:
        """Add one detection and return the updated estimate."""

        values = (
            detection.x_px,
            detection.y_px,
            detection.timestamp_s,
            detection.confidence,
            detection.width_px,
            detection.height_px,
        )
        if (
            any(not _finite(value) for value in values)
            or not 0.0 <= detection.confidence <= 1.0
            or detection.width_px < 0.0
            or detection.height_px < 0.0
        ):
            raise ValueError("detection fields must be finite and within bounds")

        if self._state_timestamp_s is None:
            self._x.initialize(detection.x_px)
            self._y.initialize(detection.y_px)
        else:
            dt = detection.timestamp_s - self._state_timestamp_s
            if dt <= 0.0:
                raise ValueError("detections must have increasing timestamps")
            self._x.predict(dt)
            self._y.predict(dt)
            self._x.update(detection.x_px)
            self._y.update(detection.y_px)

        self._state_timestamp_s = detection.timestamp_s
        self._last_measurement_timestamp_s = detection.timestamp_s
        self._target_width_px = detection.width_px
        self._target_height_px = detection.height_px
        return self._estimate(age_s=0.0)

    def predict(self, timestamp_s: float) -> Optional[TrackEstimate]:
        """Predict without a new detection."""

        if not _finite(timestamp_s):
            raise ValueError("prediction timestamp must be finite")
        if self._state_timestamp_s is None:
            return None
        age_s = timestamp_s - self._last_measurement_timestamp_s
        if age_s < 0.0:
            raise ValueError("prediction timestamp must not precede the last measurement")
        if age_s > self.max_prediction_age_s:
            self._state_timestamp_s = None
            self._last_measurement_timestamp_s = None
            return None
        dt = timestamp_s - self._state_timestamp_s
        if dt > 0.0:
            self._x.predict(dt)
            self._y.predict(dt)
            self._state_timestamp_s = timestamp_s
        return self._estimate(age_s=age_s)

    def _estimate(self, age_s: float) -> TrackEstimate:
        predicted_x = self._x.position + self._x.velocity * self.prediction_horizon_s
        predicted_y = self._y.position + self._y.velocity * self.prediction_horizon_s
        return TrackEstimate(
            x_px=self._x.position,
            y_px=self._y.position,
            vx_px_s=self._x.velocity,
            vy_px_s=self._y.velocity,
            predicted_x_px=predicted_x,
            predicted_y_px=predicted_y,
            age_s=age_s,
            target_width_px=self._target_width_px,
            target_height_px=self._target_height_px,
        )
