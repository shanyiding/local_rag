from __future__ import annotations

import json
from typing import Iterator

import requests


class OllamaError(RuntimeError):
    """Raised for any problem talking to the local Ollama server."""


class OllamaClient:
    """Minimal client for the local Ollama HTTP API (no data leaves localhost)."""

    def __init__(self, host: str = "http://localhost:11434", timeout: int = 600):
        self.host = host.rstrip("/")
        self.timeout = timeout

    # -- internals ---------------------------------------------------------
    def _post(self, path: str, payload: dict, stream: bool = False):
        try:
            r = requests.post(
                f"{self.host}{path}", json=payload, timeout=self.timeout, stream=stream
            )
        except requests.ConnectionError as e:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}. Is it running? (try `ollama serve`)"
            ) from e
        if r.status_code != 200:
            try:
                msg = r.json().get("error", r.text)
            except Exception:
                msg = r.text
            raise OllamaError(f"Ollama error {r.status_code}: {msg}")
        return r

    # -- public API --------------------------------------------------------
    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}. Is it running? (try `ollama serve`)"
            ) from e
        return [m["name"] for m in r.json().get("models", [])]

    def missing_models(self, *needed: str) -> list[str]:
        installed = set(self.list_models())
        return [m for m in needed if m not in installed and f"{m}:latest" not in installed]

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        r = self._post("/api/embed", {"model": model, "input": texts})
        return r.json()["embeddings"]

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.2,
        num_ctx: int = 4096,
    ) -> Iterator[str]:
        """Stream the assistant reply token-by-token."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        r = self._post("/api/chat", payload, stream=True)
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise OllamaError(data["error"])
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break

    def chat_text(self, *args, **kwargs) -> str:
        return "".join(self.chat(*args, **kwargs))
