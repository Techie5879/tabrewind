from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class VectorizationClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    def _post_json(self, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            url=f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama-server request failed with HTTP {exc.code}: {details}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to connect to llama-server: {exc.reason}") from exc

        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise RuntimeError("llama-server returned unexpected JSON response shape")
        return parsed

    def encode_text(self, text: str) -> list[float]:
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "input": texts,
            "model": self.model,
            "encoding_format": "float",
        }
        data = self._post_json("/v1/embeddings", payload)
        items = data.get("data")
        if not isinstance(items, list):
            raise RuntimeError("llama-server response did not include embeddings data")

        vectors: list[list[float] | None] = [None] * len(texts)
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                raise RuntimeError("llama-server response had unexpected embeddings entry")
            index_raw = item.get("index", position)
            try:
                index = int(index_raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("llama-server response returned invalid embedding index") from exc

            if index < 0 or index >= len(vectors):
                raise RuntimeError("llama-server response index out of expected range")

            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("llama-server response missing embedding vector")
            vectors[index] = [float(value) for value in embedding]

        if any(vector is None for vector in vectors):
            raise RuntimeError("llama-server response omitted one or more embeddings")

        return [vector for vector in vectors if vector is not None]


def fetch_server_slots(base_url: str, timeout_seconds: float) -> int | None:
    request = Request(url=f"{base_url.rstrip('/')}/slots", method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    if isinstance(data, list):
        return max(1, len(data))
    elif isinstance(data, dict):
        total_slots = data.get("total_slots")
        try:
            return max(1, int(total_slots))
        except (TypeError, ValueError):
            return None
    return None


__all__ = ["VectorizationClient", "fetch_server_slots"]
