"""Talk to one local Ollama server."""

import json
import math
from numbers import Real
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


MAX_OUTPUT_TOKENS = 64


class OllamaClient:
    """Small standard-library client for one local Ollama server."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout_s: float = 60.0):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Ollama URL must not be empty")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, Real)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
        ):
            raise ValueError("Ollama timeout must be positive")
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_s = float(timeout_s)

    def _request(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = urljoin(self.base_url, path.lstrip("/"))
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed ({error.code}): {detail}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Ollama is unavailable at {self.base_url}") from error

    def check(self):
        """Fail early when the local server is not running."""

        self._request("/api/tags")

    def preload(self, model: str):
        """Load one model without generating a response."""

        if not isinstance(model, str) or not model.strip():
            raise ValueError("Ollama model must not be empty")
        self._request(
            "/api/generate",
            {
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": "5m",
                "options": {"num_predict": 1},
            },
        )

    def chat(
        self,
        model: str,
        prompt: str,
        schema: Dict[str, Any],
        image: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Ollama model must not be empty")
        message: Dict[str, Any] = {"role": "user", "content": prompt}
        if image is not None:
            message["images"] = [image]
        payload = {
            "model": model,
            "messages": [message],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": MAX_OUTPUT_TOKENS},
            "keep_alive": "5m",
        }
        if think is not None:
            payload["think"] = think
        response = self._request("/api/chat", payload)
        try:
            message = response["message"]
            contents = (message.get("content"), message.get("thinking"))
            for content in contents:
                if not isinstance(content, str) or not content.strip():
                    continue
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    continue
        except (AttributeError, KeyError, TypeError) as error:
            raise RuntimeError("Ollama returned invalid structured output") from error
        raise RuntimeError("Ollama returned invalid structured output")
