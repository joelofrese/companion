"""Read optional typed dialogue without blocking flight control."""

import queue
import sys
import threading
from typing import Optional


class DialogueInput:
    """Keep typed lines available to the next conscious thought."""

    def __init__(self):
        self._messages = queue.SimpleQueue()

    def start(self):
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        print("Dialogue is live. Type follow me, hover, or stop.", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if line:
                self._messages.put(line)

    def next(self) -> Optional[str]:
        try:
            return self._messages.get_nowait()
        except queue.Empty:
            return None
