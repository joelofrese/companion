"""Conservative transcript-to-state mapping for cognitive voice commands."""

import re
from typing import Optional

from control.state_machine import State


def parse_intent(transcript: str) -> Optional[State]:
    """Map an untrusted transcript to a state, or reject it as ambiguous."""

    words = re.sub(r"[^a-z0-9 ]", " ", transcript.lower()).split()
    if not words:
        return None
    text = " ".join(words)

    if any(phrase in text for phrase in ("stop", "hover", "hold", "wait", "stay")):
        return State.HOVERING
    if any(phrase in text for phrase in ("follow", "come with me", "come along")):
        return State.FOLLOWING
    if any(phrase in text for phrase in ("respond", "face me", "turn to me", "look at me")):
        return State.RESPONDING
    if any(phrase in text for phrase in ("idle", "sleep")):
        return State.IDLE
    return None
