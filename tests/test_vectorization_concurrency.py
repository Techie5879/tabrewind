from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import VectorizationClient, fetch_server_slots


DEFAULT_TEST_MAX_WORKERS = 20


def build_client() -> tuple[VectorizationClient, str]:
    base_url = os.environ.get("TABREWIND_SMOKE_BASE_URL", "http://127.0.0.1:8080")
    model = os.environ.get("TABREWIND_SMOKE_MODEL", "embeddinggemma")
    api_key = os.environ.get("TABREWIND_SMOKE_API_KEY", "no-key")
    client = VectorizationClient(
        base_url=base_url,
        model=model,
        timeout_seconds=20.0,
        api_key=api_key,
    )
    return client, base_url


def _run_vectorization_with_workers(
    client: VectorizationClient,
    sentences: list[str],
    workers: int,
) -> tuple[list[list[float]], float]:
    started = time.perf_counter()
    if workers <= 0:
        vectors = [client.encode_text(sentence) for sentence in sentences]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(client.encode_text, sentence) for sentence in sentences]
            vectors = [future.result() for future in futures]
    elapsed_seconds = time.perf_counter() - started
    return vectors, elapsed_seconds


def _benchmark_worker_sweep(
    client: VectorizationClient,
    sentences: list[str],
    max_workers: int,
) -> dict[str, object]:
    baseline_vectors, baseline_seconds = _run_vectorization_with_workers(
        client,
        sentences,
        workers=0,
    )
    timings: list[tuple[int, float]] = [(0, baseline_seconds)]
    for workers in range(1, max_workers + 1):
        vectors, elapsed_seconds = _run_vectorization_with_workers(client, sentences, workers)
        if len(vectors) != len(baseline_vectors):
            raise RuntimeError("Mismatched vector counts during worker sweep")
        timings.append((workers, elapsed_seconds))
    return {
        "sentence_count": len(sentences),
        "baseline_seconds": baseline_seconds,
        "vector_dimensions": len(baseline_vectors[0]),
        "timings": timings,
    }


def _load_benchmark_max_workers() -> int:
    configured = DEFAULT_TEST_MAX_WORKERS
    env_value = os.environ.get("TABREWIND_TEST_MAX_WORKERS")
    if env_value:
        try:
            configured = int(env_value)
        except ValueError:
            configured = DEFAULT_TEST_MAX_WORKERS

    return max(0, configured)


def test_openai_embeddings_two_inputs(client: VectorizationClient) -> None:
    vectors = client.encode_many(["hello world", "tabrewind smoke test"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 8
    assert len(vectors[0]) == len(vectors[1])


def test_worker_sweep_zero_to_env_max(
    client: VectorizationClient,
    base_url: str,
) -> None:
    max_workers = _load_benchmark_max_workers()
    sentences = [
        f"tabrewind live benchmark sentence {index} token {(index * 19) % 29}"
        for index in range(20)
    ]
    result = _benchmark_worker_sweep(client, sentences=sentences, max_workers=max_workers)

    sentence_count = int(result["sentence_count"])
    timings = result["timings"]
    vector_dimensions = int(result["vector_dimensions"])
    baseline_seconds = float(result["baseline_seconds"])

    assert sentence_count == 20
    assert isinstance(timings, list)
    assert len(timings) == (max_workers + 1)
    assert [workers for workers, _ in timings] == list(range(max_workers + 1))
    assert vector_dimensions > 8
    assert baseline_seconds > 0.0
    for _, elapsed_seconds in timings:
        assert elapsed_seconds > 0.0

    slots = fetch_server_slots(base_url, timeout_seconds=5.0)
    if slots is not None and slots > 1:
        faster_parallel_runs = [
            (workers, elapsed)
            for workers, elapsed in timings[1:]
            if elapsed < baseline_seconds
        ]
        assert faster_parallel_runs


def run_tests() -> int:
    client, base_url = build_client()
    try:
        probe = client.encode_text("tabrewind smoke probe")
    except Exception as exc:
        print(f"SKIP: llama-server not available for tests: {exc}")
        return 0

    if not probe:
        print("SKIP: llama-server returned an empty embedding for probe")
        return 0

    test_openai_embeddings_two_inputs(client)
    test_worker_sweep_zero_to_env_max(client, base_url)
    print("OK: vectorization tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_tests())
