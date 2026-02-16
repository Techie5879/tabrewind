from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import AppConfig, VectorizationClient, chunk_entries, fetch_server_slots, load_config


DEFAULT_HEALTH_SENTENCE_COUNT = 20


def _load_int_env(key: str, default: int, minimum: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return max(minimum, parsed)


def _load_app_config() -> AppConfig:
    config_path = Path(os.environ.get("TABREWIND_CONFIG_PATH", "config.toml"))
    if config_path.exists():
        return load_config(config_path)
    return AppConfig()


def build_client(config: AppConfig) -> tuple[VectorizationClient, str]:
    client = VectorizationClient(
        base_url=config.llama.base_url,
        model=config.llama.model,
        timeout_seconds=config.llama.request_timeout_seconds,
        api_key=config.llama.api_key,
    )
    return client, config.llama.base_url


def _resolve_effective_embedding_workers(config: AppConfig, base_url: str) -> int:
    configured_workers = max(1, int(config.ingest.embedding_workers))
    slots = fetch_server_slots(
        base_url=base_url,
        timeout_seconds=config.llama.request_timeout_seconds,
    )
    if slots is None:
        return configured_workers
    return max(1, min(configured_workers, slots))


def _build_health_sentences(sentence_count: int) -> list[str]:
    return [
        f"tabrewind health sentence {index} token {(index * 19) % 29}"
        for index in range(sentence_count)
    ]


def test_openai_embeddings_two_inputs(client: VectorizationClient) -> None:
    vectors = client.encode_many(["hello world", "tabrewind health test"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 8
    assert len(vectors[0]) == len(vectors[1])


def test_batched_embeddings_health(client: VectorizationClient, config: AppConfig) -> None:
    sentence_count = _load_int_env(
        "TABREWIND_TEST_SENTENCE_COUNT",
        DEFAULT_HEALTH_SENTENCE_COUNT,
        minimum=1,
    )
    batch_size = max(1, int(config.ingest.embedding_batch_size))
    sentences = _build_health_sentences(sentence_count)

    total_vectors = 0
    vector_dimensions = 0
    for _, batch in chunk_entries(sentences, batch_size):
        vectors = client.encode_many(batch)
        if vectors and vector_dimensions == 0:
            vector_dimensions = len(vectors[0])
        total_vectors += len(vectors)

    assert total_vectors == sentence_count
    assert vector_dimensions > 8


def test_parallel_embeddings_health(
    client: VectorizationClient,
    base_url: str,
    config: AppConfig,
) -> None:
    effective_workers = _resolve_effective_embedding_workers(config, base_url)
    if effective_workers <= 1:
        return

    sentence_count = max(
        2,
        _load_int_env(
            "TABREWIND_TEST_PARALLEL_SENTENCE_COUNT",
            DEFAULT_HEALTH_SENTENCE_COUNT,
            minimum=2,
        ),
    )
    batch_size = max(1, int(config.ingest.embedding_batch_size))
    sentences = _build_health_sentences(sentence_count)
    chunks = chunk_entries(sentences, batch_size)

    total_vectors = 0
    vector_dimensions = 0
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(client.encode_many, batch): start for start, batch in chunks
        }
        for future in as_completed(futures):
            vectors = future.result()
            if vectors and vector_dimensions == 0:
                vector_dimensions = len(vectors[0])
            total_vectors += len(vectors)

    assert total_vectors == sentence_count
    assert vector_dimensions > 8


def run_tests() -> int:
    config = _load_app_config()
    client, base_url = build_client(config)
    probe = client.encode_text("tabrewind health probe")

    if not probe:
        print("SKIP: llama-server returned an empty embedding for probe")
        return 0

    test_openai_embeddings_two_inputs(client)
    test_batched_embeddings_health(client, config)
    test_parallel_embeddings_health(client, base_url, config)
    print("OK: vectorization health tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_tests())
