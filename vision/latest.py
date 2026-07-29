"""Run vision inference away from the control loop."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Optional

from control.tracking import TrackEstimate


class LatestVisionPipeline:
    """Run one inference at a time and return the newest estimate."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        self._submitted_timestamp_s: Optional[float] = None
        self._estimate: Optional[TrackEstimate] = None
        self._estimate_timestamp_s: Optional[float] = None
        self._closed = False

    def process(self, frame: Any, timestamp_s: float) -> Optional[TrackEstimate]:
        """Submit a frame when idle and return the newest estimate."""

        self._collect()
        if frame is not None and self._future is None and not self._closed:
            self._future = self._executor.submit(self.pipeline.process, frame, timestamp_s)
            self._submitted_timestamp_s = timestamp_s

        if self._estimate is None or self._estimate_timestamp_s is None:
            return None
        age_s = self._estimate.age_s + max(0.0, timestamp_s - self._estimate_timestamp_s)
        return replace(self._estimate, age_s=age_s)

    def _collect(self):
        if self._future is None or not self._future.done():
            return
        future = self._future
        self._future = None
        try:
            estimate = future.result()
        except Exception as error:
            self._estimate = None
            self._estimate_timestamp_s = None
            raise RuntimeError("vision inference failed") from error
        self._estimate = estimate
        self._estimate_timestamp_s = self._submitted_timestamp_s

    def close(self):
        """Stop accepting new frames."""

        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
