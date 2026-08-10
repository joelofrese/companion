"""Deterministic scene, intent, and fault fixtures for world simulation."""

import math
import time
from dataclasses import dataclass
from typing import Optional

from control.mind import ConsciousDecision, Telemetry, VisualObservation
from control.safety_limits import OBSTACLE_STOP_M
from control.velocity import VelocityCommand
from sim.offboard_control import (
    SECOND_FOLLOW_END_S,
    SECOND_FOLLOW_START_S,
    THIRD_FOLLOW_END_S,
    THIRD_FOLLOW_START_S,
)
from voice.intent import parse_focus, parse_intent


TARGET_RIGHT_START_S = 1.0
TARGET_RIGHT_END_S = 2.0
CONTROL_PAUSE_START_S = 1.5
CONTROL_PAUSE_END_S = 1.75
TARGET_LOST_START_S = 2.0
TARGET_LOST_END_S = 2.6
OBSTACLE_START_S = 2.8
OBSTACLE_END_S = 3.7
RECOVERY_END_S = 4.2
INVALID_SENSOR_START_S = 4.2
INVALID_SENSOR_END_S = 4.6
DROPOUT_START_S = 4.6
DROPOUT_END_S = 5.2
LINK_RECOVERY_END_S = 5.6
INVALID_COMMAND_START_S = 5.6
INVALID_COMMAND_END_S = 5.95
LOW_CONFIDENCE_START_S = 12.4
LOW_CONFIDENCE_END_S = 12.8
STALE_SENSOR_START_S = 15.0
STALE_SENSOR_END_S = 15.5
VELOCITY_TELEMETRY_START_S = 13.1
VELOCITY_TELEMETRY_END_S = 13.6
HOVER_START_S = 6.0
VISUAL_FAILURE_START_S = 12.2
VISUAL_FAILURE_END_S = 12.6
CONSCIOUS_FAILURE_START_S = 24.2
# Leave time for the independent conscious loop to observe the fault and make
# the brain command goes to zero before recovery is allowed.
CONSCIOUS_FAILURE_END_S = 25.0
MALFORMED_CONSCIOUS_START_S = 26.0
MALFORMED_CONSCIOUS_END_S = 26.6
BRAIN_SHUTDOWN_START_S = 29.0
MAX_EXPLORATORY_SPEED_M_S = 1.0
MAX_HEADING_CHANGE_DEG = 15.0
DEFAULT_EXPLORATORY_INTENT = "explore the surroundings"
NO_OBSTACLE_DISTANCE_M = 10.0


@dataclass(frozen=True)
class WorldStep:
    obstacle_distance_m: Optional[float] = NO_OBSTACLE_DISTANCE_M
    distance_fresh: bool = True
    velocity_fresh: bool = True
    transmit: bool = True
    command_override: Optional[VelocityCommand] = None
    brain_shutdown: bool = False


class SyntheticWorld:
    """Provide fixed targets, sensors, and link faults."""

    def __init__(self, exploratory: bool = False, faults: bool = False):
        self.exploratory = exploratory
        self.faults = faults

    def target_offset_east(self, elapsed_s: float) -> Optional[float]:
        if TARGET_LOST_START_S <= elapsed_s < TARGET_LOST_END_S:
            return None
        return 0.8 if TARGET_RIGHT_START_S <= elapsed_s < TARGET_RIGHT_END_S else 0.0

    def vision_confidence(self, elapsed_s: float) -> float:
        if (
            not self.exploratory
            and LOW_CONFIDENCE_START_S <= elapsed_s < LOW_CONFIDENCE_END_S
        ):
            return 0.0
        return 1.0

    def visual_failure(self, elapsed_s: float) -> bool:
        return (
            not self.exploratory
            and VISUAL_FAILURE_START_S <= elapsed_s < VISUAL_FAILURE_END_S
        )

    def step(self, elapsed_s: float) -> WorldStep:
        if self.exploratory and not self.faults:
            return WorldStep()
        if elapsed_s >= BRAIN_SHUTDOWN_START_S:
            return WorldStep(brain_shutdown=True)
        if STALE_SENSOR_START_S <= elapsed_s < STALE_SENSOR_END_S:
            return WorldStep(distance_fresh=False)
        if VELOCITY_TELEMETRY_START_S <= elapsed_s < VELOCITY_TELEMETRY_END_S:
            return WorldStep(velocity_fresh=False)
        if OBSTACLE_START_S <= elapsed_s < OBSTACLE_END_S:
            return WorldStep(obstacle_distance_m=0.3)
        if OBSTACLE_END_S <= elapsed_s < RECOVERY_END_S:
            return WorldStep()
        if INVALID_SENSOR_START_S <= elapsed_s < INVALID_SENSOR_END_S:
            return WorldStep(obstacle_distance_m=math.nan)
        if DROPOUT_START_S <= elapsed_s < DROPOUT_END_S:
            return WorldStep(transmit=False)
        if DROPOUT_END_S <= elapsed_s < LINK_RECOVERY_END_S:
            return WorldStep()
        if INVALID_COMMAND_START_S <= elapsed_s < INVALID_COMMAND_END_S:
            return WorldStep(
                command_override=VelocityCommand(
                    forward_m_s=1.0,
                    right_m_s=1.0,
                    down_m_s=1.0,
                    yaw_rate_deg_s=90.0,
                )
            )
        return WorldStep()


class WorldVisualModel:
    """Provide fixed scene descriptions to the visual boundary."""

    def __init__(
        self,
        world: SyntheticWorld,
        started_at_s: float,
        synthetic_scene: bool = True,
    ):
        self.world = world
        self.started_at_s = started_at_s
        self.synthetic_scene = synthetic_scene

    def observe(
        self,
        _image,
        timestamp_s: float,
        focus: str,
        intent: str,
        previous_movement: str,
        previous_observation: str,
        telemetry: Telemetry,
    ) -> VisualObservation:
        if not self.synthetic_scene:
            return VisualObservation(
                timestamp_s=timestamp_s,
                description="camera frame received; no visual model configured",
                next_focus=focus,
            )
        elapsed_s = max(0.0, timestamp_s - self.started_at_s)
        if self.world.visual_failure(elapsed_s):
            raise RuntimeError("simulated visual model failure")
        target_offset_east_m = self.world.target_offset_east(elapsed_s)
        intent_kind = parse_intent(intent)
        focus_text = " ".join(focus.lower().split())
        focused_person = focus_text == "person"
        following = intent_kind == "following"
        exploring = intent_kind == "exploring"
        tracking_person = following or focused_person
        if target_offset_east_m is None:
            description = (
                "no person is visible"
                if tracking_person
                else "no clear path is visible"
            )
            movement = "stop"
        elif target_offset_east_m > 0.0:
            movement = "right"
            description = (
                "the person is to the right"
                if tracking_person
                else "open space is to the right"
            )
        else:
            movement = "forward"
            description = (
                "the person is ahead" if tracking_person else "open space is ahead"
            )
        if not (following or exploring or focused_person):
            movement = "stop"
        description_text = " ".join(description.lower().split())
        focused_answer = (
            description
            if focus_text
            and focus_text in description_text
            and f"no {focus_text}" not in description_text
            else ""
        )
        alternate_movement = ""
        if (
            movement in ("forward", "stop")
            and telemetry.obstacle_distance_m is not None
            and math.isfinite(telemetry.obstacle_distance_m)
            and telemetry.obstacle_distance_m <= OBSTACLE_STOP_M
        ):
            alternate_movement = "right"
        return VisualObservation(
            timestamp_s=timestamp_s,
            description=description,
            focused_answer=focused_answer,
            movement=movement,
            alternate_movement=alternate_movement,
            next_focus=focus or ("person" if following else ""),
            confidence=self.world.vision_confidence(elapsed_s),
        )


class WorldLanguageModel:
    """Choose simulated intent while exercising the conscious boundary."""

    def __init__(self, exploratory: bool):
        self.exploratory = exploratory
        self.started_at_s = 0.0
        self.intent = None
        self._conscious_failure_seen = False
        self.recovered_with_observation = False
        self.malformed_conscious_seen = False

    def think(self, information) -> ConsciousDecision:
        if self.intent is None:
            self.intent = information.intent
        previous_intent = self.intent
        dialogue = ""
        if information.dialogue:
            intent = parse_intent(information.dialogue)
            if intent is None:
                focus = parse_focus(information.dialogue)
                dialogue = (
                    f"Looking for {focus}."
                    if focus
                    else "I did not understand that request."
                )
            else:
                self.intent = intent
                dialogue = f"Intent changed to {self.intent}."
        if not self.exploratory:
            elapsed_s = max(0.0, time.monotonic() - self.started_at_s)
            if CONSCIOUS_FAILURE_START_S <= elapsed_s < CONSCIOUS_FAILURE_END_S:
                self._conscious_failure_seen = True
                raise RuntimeError("simulated conscious model failure")
            if self._conscious_failure_seen and not self.recovered_with_observation:
                if not information.new_observations:
                    raise RuntimeError(
                        "simulated conscious recovery lost visual context"
                    )
                self.recovered_with_observation = True
            if MALFORMED_CONSCIOUS_START_S <= elapsed_s < MALFORMED_CONSCIOUS_END_S:
                self.malformed_conscious_seen = True
                return ConsciousDecision(intent="", intent_changed=True)
            following = (
                elapsed_s < HOVER_START_S
                or SECOND_FOLLOW_START_S <= elapsed_s < SECOND_FOLLOW_END_S
                or THIRD_FOLLOW_START_S <= elapsed_s < THIRD_FOLLOW_END_S
            )
            self.intent = "following" if following else "hover"
        summary = information.summary
        if information.new_observations:
            summary = information.new_observations[-1].description
        return ConsciousDecision(
            intent=self.intent,
            intent_changed=self.intent != previous_intent,
            focus="person" if parse_intent(self.intent) == "following" else "",
            dialogue=dialogue,
            summary=summary or "The simulated world is running.",
        )
