"""Temporary conscious model used until a real LLM is connected."""

from control.mind import ConsciousDecision


class IntentLanguageModel:
    """Preserve the current intent while the brain backend is being chosen."""

    def think(self, information) -> ConsciousDecision:
        intent = information.intent
        summary = information.summary
        if information.new_observations:
            summary = information.new_observations[-1].description
        return ConsciousDecision(
            intent=intent,
            focus="person" if intent == "following" else "",
            summary=summary or "The fallback brain is running.",
        )
