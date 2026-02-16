from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import AppConfig, VectorizationClient, chunk_entries, fetch_server_slots, load_config


DEFAULT_HEALTH_SENTENCE_COUNT = 20
DEFAULT_HEALTH_SEED = 1337
DEFAULT_REPORT_SENTENCE_COUNT = 5_000
DEFAULT_REPORT_SEED = 1337


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


def build_client(config: AppConfig | None = None) -> tuple[VectorizationClient, str, AppConfig]:
    resolved = _load_app_config() if config is None else config
    client = VectorizationClient(
        base_url=resolved.llama.base_url,
        model=resolved.llama.model,
        timeout_seconds=resolved.llama.request_timeout_seconds,
        api_key=resolved.llama.api_key,
    )
    return client, resolved.llama.base_url, resolved


def _resolve_effective_embedding_workers(config: AppConfig, base_url: str) -> tuple[int, int | None]:
    configured_workers = max(1, int(config.ingest.embedding_workers))
    slots = fetch_server_slots(
        base_url=base_url,
        timeout_seconds=config.llama.request_timeout_seconds,
    )
    if slots is None:
        return configured_workers, None
    return max(1, min(configured_workers, slots)), slots


def _load_benchmark_max_workers(default_value: int) -> int:
    fallback = max(0, default_value)
    return _load_int_env("TABREWIND_TEST_MAX_WORKERS", fallback, minimum=0)


def _generate_sentences(sentence_count: int, seed: int) -> list[str]:
    generator = random.Random(seed)
    base = [f"tabrewind sweep sentence {index} token {(index * 19) % 29}" for index in range(20)]
    return [
        f"{base[index % len(base)]} seed {generator.randint(0, 10_000_000)} idx {index}"
        for index in range(sentence_count)
    ]


def _run_batched_vectorization_with_workers(
    client: VectorizationClient,
    sentences: list[str],
    workers: int,
    batch_size: int,
) -> tuple[int, int, float]:
    started = time.perf_counter()
    chunked = chunk_entries(sentences, max(1, batch_size))
    vector_dimensions = 0
    total_vectors = 0

    if workers <= 0:
        for _, batch in chunked:
            vectors = client.encode_many(batch)
            if vectors and vector_dimensions == 0:
                vector_dimensions = len(vectors[0])
            total_vectors += len(vectors)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(client.encode_many, batch): start for start, batch in chunked
            }
            for future in as_completed(futures):
                vectors = future.result()
                if vectors and vector_dimensions == 0:
                    vector_dimensions = len(vectors[0])
                total_vectors += len(vectors)

    elapsed_seconds = time.perf_counter() - started
    return total_vectors, vector_dimensions, elapsed_seconds


def _benchmark_worker_sweep(
    client: VectorizationClient,
    sentences: list[str],
    max_workers: int,
    batch_size: int,
) -> dict[str, object]:
    baseline_count, baseline_dims, baseline_seconds = _run_batched_vectorization_with_workers(
        client=client,
        sentences=sentences,
        workers=0,
        batch_size=batch_size,
    )
    if baseline_count != len(sentences):
        raise RuntimeError("Mismatched vector counts for baseline run")

    timings: list[tuple[int, float]] = [(0, baseline_seconds)]
    for workers in range(1, max_workers + 1):
        vector_count, vector_dims, elapsed_seconds = _run_batched_vectorization_with_workers(
            client=client,
            sentences=sentences,
            workers=workers,
            batch_size=batch_size,
        )
        if vector_count != baseline_count:
            raise RuntimeError("Mismatched vector counts during worker sweep")
        if vector_dims != baseline_dims:
            raise RuntimeError("Mismatched vector dimensions during worker sweep")
        timings.append((workers, elapsed_seconds))

    return {
        "sentence_count": len(sentences),
        "baseline_seconds": baseline_seconds,
        "vector_dimensions": baseline_dims,
        "timings": timings,
    }


def _plot_sweep_png(result: dict[str, object], output_path: Path) -> None:
    timings = result["timings"]
    assert isinstance(timings, list)

    workers = [int(workers) for workers, _ in timings]
    seconds = [float(elapsed) for _, elapsed in timings]

    baseline_seconds = float(result["baseline_seconds"])
    best_index = min(range(len(seconds)), key=lambda idx: seconds[idx])
    best_workers = workers[best_index]
    best_seconds = seconds[best_index]

    plt.figure(figsize=(11, 6))
    plt.plot(workers, seconds, marker="o", linewidth=2, color="#2563eb")
    plt.scatter([0], [baseline_seconds], color="#7c3aed", s=70, zorder=4)
    plt.scatter([best_workers], [best_seconds], color="#dc2626", s=80, zorder=5)

    plt.title("TABREWIND Vectorization Sweep")
    plt.xlabel("Workers")
    plt.ylabel("Elapsed Seconds (lower is better)")
    plt.grid(True, alpha=0.3)
    plt.xticks(workers)

    plt.annotate(
        f"baseline 0w: {baseline_seconds:.2f}s",
        xy=(0, baseline_seconds),
        xytext=(10, 10),
        textcoords="offset points",
        color="#6d28d9",
    )
    plt.annotate(
        f"best {best_workers}w: {best_seconds:.2f}s ({baseline_seconds / best_seconds:.2f}x)",
        xy=(best_workers, best_seconds),
        xytext=(10, -14),
        textcoords="offset points",
        color="#991b1b",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_sweep_report(
    sentence_count: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    config = _load_app_config()
    client, base_url, _ = build_client(config)

    effective_workers, slots = _resolve_effective_embedding_workers(config, base_url)
    max_workers = _load_benchmark_max_workers(effective_workers)
    batch_size = max(1, int(config.ingest.embedding_batch_size))
    sentences = _generate_sentences(sentence_count, seed)

    result = _benchmark_worker_sweep(
        client=client,
        sentences=sentences,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    baseline_seconds = float(result["baseline_seconds"])
    timings = result["timings"]
    assert isinstance(timings, list)
    best_workers, best_seconds = min(timings, key=lambda row: float(row[1]))

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"vectorization_sweep_{sentence_count}.json"
    png_path = output_dir / f"vectorization_sweep_{sentence_count}.png"

    payload = {
        "base_url": base_url,
        "slots": slots,
        "configured_embedding_workers": int(config.ingest.embedding_workers),
        "effective_embedding_workers": int(effective_workers),
        "batch_size": batch_size,
        "max_workers": int(max_workers),
        "sentence_count": int(result["sentence_count"]),
        "vector_dimensions": int(result["vector_dimensions"]),
        "seed": seed,
        "baseline_seconds": baseline_seconds,
        "best_workers": int(best_workers),
        "best_seconds": float(best_seconds),
        "best_speedup_vs_baseline": baseline_seconds / float(best_seconds),
        "timings": [
            {
                "workers": int(workers),
                "seconds": float(elapsed),
                "sentences_per_second": int(result["sentence_count"]) / float(elapsed),
            }
            for workers, elapsed in timings
        ],
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _plot_sweep_png(result, png_path)

    return {
        "json_path": str(json_path),
        "png_path": str(png_path),
        "best_workers": int(best_workers),
        "best_seconds": float(best_seconds),
        "baseline_seconds": baseline_seconds,
    }


def test_openai_embeddings_two_inputs(client: VectorizationClient) -> None:
    vectors = client.encode_many(["hello world", "tabrewind health test"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 8
    assert len(vectors[0]) == len(vectors[1])


def test_worker_sweep_zero_to_runtime_max(
    client: VectorizationClient,
    base_url: str,
    config: AppConfig,
) -> None:
    effective_workers, slots = _resolve_effective_embedding_workers(config, base_url)
    max_workers = _load_benchmark_max_workers(effective_workers)
    sentence_count = _load_int_env(
        "TABREWIND_TEST_SENTENCE_COUNT",
        DEFAULT_HEALTH_SENTENCE_COUNT,
        minimum=1,
    )
    seed = _load_int_env("TABREWIND_TEST_SENTENCE_SEED", DEFAULT_HEALTH_SEED, minimum=0)
    batch_size = max(1, int(config.ingest.embedding_batch_size))
    sentences = _generate_sentences(sentence_count=sentence_count, seed=seed)

    result = _benchmark_worker_sweep(
        client=client,
        sentences=sentences,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    observed_count = int(result["sentence_count"])
    timings = result["timings"]
    vector_dimensions = int(result["vector_dimensions"])
    baseline_seconds = float(result["baseline_seconds"])

    assert observed_count == sentence_count
    assert isinstance(timings, list)
    assert len(timings) == (max_workers + 1)
    assert [workers for workers, _ in timings] == list(range(max_workers + 1))
    assert vector_dimensions > 8
    assert baseline_seconds > 0.0
    for _, elapsed_seconds in timings:
        assert elapsed_seconds > 0.0

    capped_parallel_max = min(max_workers, effective_workers)
    if slots is not None and slots > 1 and capped_parallel_max > 0:
        faster_parallel_runs = [
            (workers, elapsed)
            for workers, elapsed in timings[1 : capped_parallel_max + 1]
            if elapsed < baseline_seconds
        ]
        assert faster_parallel_runs


def run_tests() -> int:
    config = _load_app_config()
    client, base_url, _ = build_client(config)
    probe = client.encode_text("tabrewind health probe")

    if not probe:
        print("SKIP: llama-server returned an empty embedding for probe")
        return 0

    test_openai_embeddings_two_inputs(client)
    test_worker_sweep_zero_to_runtime_max(client, base_url, config)
    print("OK: vectorization tests passed")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vectorization-concurrency-tests",
        description="Run health tests or generate a sweep report.",
    )
    parser.add_argument(
        "--sweep-report",
        action="store_true",
        help="Generate a JSON + PNG sweep report using runtime-parity batching/workers.",
    )
    parser.add_argument(
        "--sentence-count",
        type=int,
        default=DEFAULT_REPORT_SENTENCE_COUNT,
        help="Sentence count for sweep report generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_REPORT_SEED,
        help="Random seed used when generating sweep sentences.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Output directory for report files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sweep_report:
        result = write_sweep_report(
            sentence_count=max(1, args.sentence_count),
            seed=max(0, args.seed),
            output_dir=args.output_dir,
        )
        print(json.dumps(result))
        return 0
    return run_tests()


if __name__ == "__main__":
    raise SystemExit(main())
