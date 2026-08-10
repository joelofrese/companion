"""Turn voice transcripts into safe intents."""

import re
from typing import Optional


FOCUS_PHRASES = (
    "follow",
    "look for",
    "look at",
    "search for",
    "find",
    "locate",
    "inspect",
    "identify",
    "spot",
    "show me",
    "where is",
)
FOCUS_STOP_WORDS = frozenset(("around", "in", "near", "on", "over", "under"))
FOCUS_CONNECTORS = frozenset(("and", "then", "while"))
FOCUS_ACTION_WORDS = frozenset(
    (
        "find",
        "follow",
        "fly",
        "go",
        "inspect",
        "locate",
        "look",
        "move",
        "search",
        "show",
        "spot",
        "stop",
        "stay",
        "wander",
        "where",
    )
)
NEGATION_WORDS = frozenset(("no", "not", "never", "dont", "don't"))
MOTION_WORDS = frozenset(
    ("move", "follow", "fly", "go", "come", "explore", "wander", "search")
)


def _words(transcript: str) -> list[str]:
    if not isinstance(transcript, str):
        return []
    normalized = transcript.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9']", " ", normalized).split()


def _contains_phrase(words: list[str], phrase: str) -> bool:
    phrase_words = phrase.split()
    width = len(phrase_words)
    return any(
        words[index:index + width] == phrase_words
        for index in range(len(words) - width + 1)
    )


def _focus_boundary(words: list[str], position: int) -> bool:
    word = words[position]
    if word in FOCUS_STOP_WORDS:
        return True
    if word not in FOCUS_CONNECTORS:
        return False
    position += 1
    while position < len(words) and words[position] in FOCUS_CONNECTORS:
        position += 1
    return position < len(words) and words[position] in FOCUS_ACTION_WORDS


def parse_focus(transcript: str) -> Optional[str]:
    """Return the subject of a direct visual request, if there is one."""

    words = _words(transcript)
    if not words or any(word in NEGATION_WORDS for word in words):
        return None
    for phrase in FOCUS_PHRASES:
        phrase_words = phrase.split()
        width = len(phrase_words)
        for index in range(len(words) - width + 1):
            if words[index:index + width] != phrase_words:
                continue
            subject = words[index + width:]
            while subject and subject[0] in {"a", "an", "my", "that", "the"}:
                subject.pop(0)
            stop = next(
                (
                    position
                    for position in range(len(subject))
                    if _focus_boundary(subject, position)
                ),
                len(subject),
            )
            subject = subject[:stop]
            subject = [word for word in subject if word != "please"][:4]
            if subject:
                if subject in (["me"], ["us"]):
                    return "person"
                return " ".join(subject)
    return None


def parse_intent(transcript: str) -> Optional[str]:
    """Return one clear intent, or none."""

    words = _words(transcript)
    if not words:
        return None

    if any(word in NEGATION_WORDS for word in words):
        if any(word in MOTION_WORDS for word in words):
            return "hover"
        return None

    if _contains_phrase(words, "stay with me"):
        return "following"

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
        "exploring": (
            "explore",
            "exploring",
            "look around",
            "search",
            "wander",
        ),
        "hover": ("idle", "sleep"),
    }
    matches = [
        intent
        for intent, phrases in intents.items()
        if any(_contains_phrase(words, phrase) for phrase in phrases)
    ]
    return matches[0] if len(matches) == 1 else None
