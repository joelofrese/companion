"""Turn voice transcripts into safe intents."""

import re
from typing import Optional


def _contains_phrase(words: list[str], phrase: str) -> bool:
    phrase_words = phrase.split()
    width = len(phrase_words)
    return any(
        words[index:index + width] == phrase_words
        for index in range(len(words) - width + 1)
    )


def parse_intent(transcript: str) -> Optional[str]:
    """Return one clear intent, or none."""

    if not isinstance(transcript, str):
        return None
    normalized = transcript.lower().replace("’", "'")
    words = re.sub(r"[^a-z0-9']", " ", normalized).split()
    if not words:
        return None

    if any(word in {"no", "not", "never", "dont", "don't"} for word in words):
        return None

    if any(
        _contains_phrase(words, phrase)
        for phrase in (
            "stop",
            "hover",
            "hold",
            "wait",
            "stay",
            "remain still",
            "maintain position",
        )
    ):
        return "hover"

    intents = {
        "following": ("follow", "following", "come with me", "come along"),
        "hover": ("idle", "sleep"),
    }
    matches = [
        intent
        for intent, phrases in intents.items()
        if any(_contains_phrase(words, phrase) for phrase in phrases)
    ]
    return matches[0] if len(matches) == 1 else None
