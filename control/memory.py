"""Keep a small, editable record of companion experience."""

from pathlib import Path


MAX_MEMORY_LINES = 64
MAX_MEMORY_CHARS = 360
MEMORY_CONTEXT_LINES = 8


class CompanionMemory:
    """Keep a bounded record of prior experience."""

    def __init__(self, path):
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ValueError("memory path must not be empty")
        self.path = Path(path).expanduser()
        self._lines = self._read()

    def _read(self):
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [
            line.strip()[:MAX_MEMORY_CHARS]
            for line in lines
            if line.strip()
        ][-MAX_MEMORY_LINES:]

    def context(self) -> str:
        """Return recent experience for the companion brain."""

        return "\n".join(self._lines[-MEMORY_CONTEXT_LINES:])

    def remember(self, entry: str):
        """Save one new memory and keep the file bounded."""

        entry = " ".join(entry.split())[:MAX_MEMORY_CHARS]
        if not entry:
            return
        if self._lines and self._lines[-1] == entry:
            return
        self._lines = (self._lines + [entry])[-MAX_MEMORY_LINES:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")
