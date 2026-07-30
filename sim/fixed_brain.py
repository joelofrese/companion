"""Keep the requested intent during a deterministic simulation."""

from control.mind import ConsciousDecision


class FixedLanguageModel:
    """Return the intent supplied by the simulation schedule."""

    def think(self, information) -> ConsciousDecision:
        summary = information.summary
        if information.new_observations:
            summary = information.new_observations[-1].description
        return ConsciousDecision(
            intent=information.intent,
            focus="person" if information.intent == "following" else "",
            summary=summary or "The simulated world is running.",
        )
