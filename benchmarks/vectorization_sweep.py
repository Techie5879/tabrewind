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


DEFAULT_SENTENCE_COUNT = 5_000
DEFAULT_SEED = 1337


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


def _build_client(config: AppConfig) -> tuple[VectorizationClient, str]:
    client = VectorizationClient(
        base_url=config.llama.base_url,
        model=config.llama.model,
        timeout_seconds=config.llama.request_timeout_seconds,
        api_key=config.llama.api_key,
    )
    return client, config.llama.base_url


def _resolve_effective_embedding_workers(config: AppConfig, base_url: str) -> tuple[int, int | None]:
    configured_workers = max(1, int(config.ingest.embedding_workers))
    slots = fetch_server_slots(
        base_url=base_url,
        timeout_seconds=config.llama.request_timeout_seconds,
    )
    if slots is None:
        return configured_workers, None
    return max(1, min(configured_workers, slots)), slots


def _build_sentences(sentence_count: int, seed: int) -> list[str]:
    generator = random.Random(seed)
    base = [
        f"tabrewind benchmark sentence {index} token {(index * 19) % 29}"
        for index in range(20)
    ]
    return [
        f"{base[index % len(base)]} seed {generator.randint(0, 10_000_000)} idx {index}"
        for index in range(sentence_count)
    ]


def _run_batched_vectorization(
    client: VectorizationClient,
    sentences: list[str],
    workers: int,
    batch_size: int,
) -> tuple[int, int, float]:
    started = time.perf_counter()
    chunks = chunk_entries(sentences, max(1, batch_size))
    vector_dimensions = 0
    total_vectors = 0

    if workers <= 0:
        for _, batch in chunks:
            vectors = client.encode_many(batch)
            if vectors and vector_dimensions == 0:
                vector_dimensions = len(vectors[0])
            total_vectors += len(vectors)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(client.encode_many, batch): start for start, batch in chunks
            }
            for future in as_completed(futures):
                vectors = future.result()
                if vectors and vector_dimensions == 0:
                    vector_dimensions = len(vectors[0])
                total_vectors += len(vectors)

    elapsed_seconds = time.perf_counter() - started
    return total_vectors, vector_dimensions, elapsed_seconds


def run_worker_sweep(
    client: VectorizationClient,
    sentences: list[str],
    max_workers: int,
    batch_size: int,
) -> dict[str, object]:
    baseline_count, baseline_dimensions, baseline_seconds = _run_batched_vectorization(
        client=client,
        sentences=sentences,
        workers=0,
        batch_size=batch_size,
    )
    if baseline_count != len(sentences):
        raise RuntimeError("Mismatched vector counts for baseline run")

    timings: list[tuple[int, float]] = [(0, baseline_seconds)]
    for workers in range(1, max_workers + 1):
        count, dimensions, elapsed_seconds = _run_batched_vectorization(
            client=client,
            sentences=sentences,
            workers=workers,
            batch_size=batch_size,
        )
        if count != baseline_count:
            raise RuntimeError("Mismatched vector counts during worker sweep")
        if dimensions != baseline_dimensions:
            raise RuntimeError("Mismatched vector dimensions during worker sweep")
        timings.append((workers, elapsed_seconds))

    return {
        "sentence_count": len(sentences),
        "vector_dimensions": baseline_dimensions,
        "baseline_seconds": baseline_seconds,
        "timings": timings,
    }


def _plot_sweep_png(result: dict[str, object], output_path: Path) -> None:
    timings = result["timings"]
    assert isinstance(timings, list)

    workers = [int(workers) for workers, _ in timings]
    seconds = [float(elapsed) for _, elapsed in timings]

    baseline_seconds = float(result["baseline_seconds"])
    best_index = min(range(len(seconds)), key=lambda index: seconds[index])
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
    max_workers_override: int | None,
) -> dict[str, object]:
    config = _load_app_config()
    client, base_url = _build_client(config)
    effective_workers, slots = _resolve_effective_embedding_workers(config, base_url)

    if max_workers_override is None:
        max_workers = _load_int_env(
            "TABREWIND_BENCH_MAX_WORKERS",
            effective_workers,
            minimum=0,
        )
    else:
        max_workers = max(0, int(max_workers_override))

    batch_size = max(1, int(config.ingest.embedding_batch_size))
    sentences = _build_sentences(sentence_count, seed)
    result = run_worker_sweep(
        client=client,
        sentences=sentences,
        max_workers=max_workers,
        batch_size=batch_size,
    )

    timings = result["timings"]
    assert isinstance(timings, list)
    baseline_seconds = float(result["baseline_seconds"])
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vectorization-sweep",
        description="Generate a worker sweep benchmark report.",
    )
    parser.add_argument(
        "--sentence-count",
        type=int,
        default=DEFAULT_SENTENCE_COUNT,
        help="Number of generated sentences to benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used when generating sentences.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional sweep upper bound; defaults to effective embedding workers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Directory where JSON and PNG outputs are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = write_sweep_report(
        sentence_count=max(1, args.sentence_count),
        seed=max(0, args.seed),
        output_dir=args.output_dir,
        max_workers_override=args.max_workers,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
