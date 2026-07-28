"""Conservative transcript-to-state mapping for cognitive voice commands."""

import re
from typing import Optional

from control.state_machine import State


def _contains_phrase(words: list[str], phrase: str) -> bool:
    phrase_words = phrase.split()
    width = len(phrase_words)
    return any(words[index:index + width] == phrase_words for index in range(len(words) - width + 1))


def parse_intent(transcript: str) -> Optional[State]:
    """Map an untrusted transcript to a state, or reject it as ambiguous."""

    if not isinstance(transcript, str):
        return None
    normalized = transcript.lower().replace("’", "'")
    words = re.sub(r"[^a-z0-9']", " ", normalized).split()
    if not words:
        return None

    if any(word in {"no", "not", "never", "dont", "don't"} for word in words):
        return None

    if any(_contains_phrase(words, phrase) for phrase in ("stop", "hover", "hold", "wait", "stay")):
        return State.HOVERING
    if any(_contains_phrase(words, phrase) for phrase in ("follow", "come with me", "come along")):
        return State.FOLLOWING
    if any(_contains_phrase(words, phrase) for phrase in ("respond", "face me", "turn to me", "look at me")):
        return State.RESPONDING
    if any(_contains_phrase(words, phrase) for phrase in ("idle", "sleep")):
        return State.IDLE
    return None
