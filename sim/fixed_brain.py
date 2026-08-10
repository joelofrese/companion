"""Keep deterministic brain behavior for simulation fixtures."""

from control.mind import ConsciousDecision, Telemetry, VisualObservation


class FixedVisualModel:
    """Return a fixed person or no-person observation for RTP checks."""

    def __init__(self, person: bool):
        self.person = person
        self.following_focus_seen = False
        self.following_focus_lost = False

    def observe(
        self,
        image,
        timestamp_s: float,
        focus: str,
        intent: str,
        previous_movement: str,
        previous_observation: str,
        telemetry: Telemetry,
    ) -> VisualObservation:
        if intent != "following":
            self.following_focus_seen = False
        elif focus:
            self.following_focus_seen = True
        elif self.following_focus_seen:
            self.following_focus_lost = True
        if self.person:
            description = "a person is visible ahead"
            movement = "forward" if intent == "following" else "stop"
            confidence = 1.0
        else:
            description = "no person is visible"
            movement = "stop"
            confidence = 0.0
        return VisualObservation(
            timestamp_s=timestamp_s,
            description=description,
            focused_answer=description if focus else "",
            movement=movement,
            next_focus=focus or "person",
            confidence=confidence,
        )


class FixedLanguageModel:
    """Return the intent supplied by the simulation schedule."""

    def think(self, information) -> ConsciousDecision:
        summary = information.summary
        if information.new_observations:
            summary = information.new_observations[-1].description
        return ConsciousDecision(
            intent=information.intent,
            intent_changed=False,
            focus="person" if information.intent == "following" else "",
            summary=summary or "The simulated world is running.",
        )
