"""Join person detection and tracking."""

from typing import Any, Optional, Protocol

from control.tracking import Detection, PersonTracker, TrackEstimate


class PersonDetector(Protocol):
    def detect(self, frame: Any, timestamp_s: float) -> Optional[Detection]:
        ...


class PersonVisionPipeline:
    """Turn frames into tracked person positions."""

    def __init__(self, detector: PersonDetector, tracker: Optional[PersonTracker] = None):
        self.detector = detector
        self.tracker = tracker or PersonTracker()

    def process(self, frame: Any, timestamp_s: float) -> Optional[TrackEstimate]:
        """Detect and track one frame."""

        if frame is None:
            return self.tracker.predict(timestamp_s)
        detection = self.detector.detect(frame, timestamp_s)
        if detection is not None:
            return self.tracker.update(detection)
        return self.tracker.predict(timestamp_s)
