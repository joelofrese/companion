"""Lightweight constant-velocity Kalman tracker for detector measurements."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Detection:
    """A person's image-plane center in pixels and its capture timestamp."""

    x_px: float
    y_px: float
    timestamp_s: float
    confidence: float = 1.0


@dataclass(frozen=True)
class TrackEstimate:
    """Filtered position, velocity, and short-horizon predicted position."""

    x_px: float
    y_px: float
    vx_px_s: float
    vy_px_s: float
    predicted_x_px: float
    predicted_y_px: float


class _AxisFilter:
    """One position/velocity Kalman filter; two instances form the 2D tracker."""

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
    """Track a detected person's center and predict a short time into the future."""

    def __init__(self, prediction_horizon_s: float = 0.3):
        if prediction_horizon_s < 0.0:
            raise ValueError("prediction horizon must not be negative")
        self.prediction_horizon_s = prediction_horizon_s
        self._x = _AxisFilter(process_noise=1.0, measurement_noise=4.0)
        self._y = _AxisFilter(process_noise=1.0, measurement_noise=4.0)
        self._last_timestamp_s: Optional[float] = None

    def update(self, detection: Detection) -> TrackEstimate:
        """Consume one detector measurement and return its filtered estimate."""

        if self._last_timestamp_s is None:
            self._x.initialize(detection.x_px)
            self._y.initialize(detection.y_px)
        else:
            dt = detection.timestamp_s - self._last_timestamp_s
            if dt <= 0.0:
                raise ValueError("detections must have increasing timestamps")
            self._x.predict(dt)
            self._y.predict(dt)
            self._x.update(detection.x_px)
            self._y.update(detection.y_px)

        self._last_timestamp_s = detection.timestamp_s
        predicted_x = self._x.position + self._x.velocity * self.prediction_horizon_s
        predicted_y = self._y.position + self._y.velocity * self.prediction_horizon_s
        return TrackEstimate(
            x_px=self._x.position,
            y_px=self._y.position,
            vx_px_s=self._x.velocity,
            vy_px_s=self._y.velocity,
            predicted_x_px=predicted_x,
            predicted_y_px=predicted_y,
        )
