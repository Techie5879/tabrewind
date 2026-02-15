from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import fetch_server_slots
from tests.test_vectorization_concurrency import build_client


def run_sweep(sentence_count: int, max_workers: int, batch_size: int, seed: int) -> dict[str, object]:
    client, base_url = build_client()

    random.seed(seed)
    base = [f"tabrewind scale sentence {index} token {(index * 19) % 29}" for index in range(20)]
    sentences = [
        f"{base[index % len(base)]} seed {random.randint(0, 10_000_000)} idx {index}"
        for index in range(sentence_count)
    ]
    batches = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]

    first_vector = client.encode_many(batches[0])[0]
    vector_dimensions = len(first_vector)
    slots = fetch_server_slots(base_url, timeout_seconds=5.0)

    timings: list[dict[str, float | int]] = []
    for workers in range(max_workers + 1):
        started = time.perf_counter()
        if workers <= 0:
            total = 0
            for batch in batches:
                total += len(client.encode_many(batch))
        else:
            total = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(client.encode_many, batch) for batch in batches]
                for future in futures:
                    total += len(future.result())

        elapsed = time.perf_counter() - started
        if total != sentence_count:
            raise RuntimeError(
                f"Mismatched embedding count for workers={workers}: {total} != {sentence_count}"
            )
        timings.append(
            {
                "workers": workers,
                "seconds": elapsed,
                "sentences_per_second": sentence_count / elapsed,
            }
        )

    baseline = float(timings[0]["seconds"])
    best = min(timings, key=lambda row: float(row["seconds"]))

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "slots": slots,
        "sentence_count": sentence_count,
        "batch_size": batch_size,
        "max_workers": max_workers,
        "random_seed": seed,
        "vector_dimensions": vector_dimensions,
        "baseline_seconds": baseline,
        "best_workers": int(best["workers"]),
        "best_seconds": float(best["seconds"]),
        "best_speedup_vs_baseline": baseline / float(best["seconds"]),
        "timings": timings,
    }


def render_svg(data: dict[str, object]) -> str:
    timings = data["timings"]
    assert isinstance(timings, list)

    width = 1100
    height = 640
    margin_left = 90
    margin_right = 40
    margin_top = 60
    margin_bottom = 120
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_workers = int(data["max_workers"])
    baseline = float(data["baseline_seconds"])
    best_workers = int(data["best_workers"])
    best_seconds = float(data["best_seconds"])
    sentence_count = int(data["sentence_count"])
    batch_size = int(data["batch_size"])
    slots = data["slots"]

    seconds_values = [float(row["seconds"]) for row in timings if isinstance(row, dict)]
    y_min = min(seconds_values) * 0.95
    y_max = max(seconds_values) * 1.05
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_to_px(x: int) -> float:
        return margin_left + (x / max_workers) * plot_w

    def y_to_px(y: float) -> float:
        return margin_top + (1.0 - ((y - y_min) / (y_max - y_min))) * plot_h

    points = " ".join(
        f"{x_to_px(int(row['workers'])):.2f},{y_to_px(float(row['seconds'])):.2f}"
        for row in timings
        if isinstance(row, dict)
    )

    best_x = x_to_px(best_workers)
    best_y = y_to_px(best_seconds)
    base_x = x_to_px(0)
    base_y = y_to_px(baseline)

    h_grid: list[tuple[float, float]] = []
    for i in range(6):
        frac = i / 5
        y = margin_top + frac * plot_h
        val = y_max - frac * (y_max - y_min)
        h_grid.append((y, val))

    x_ticks = list(range(0, max_workers + 1, 2))

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect width="100%" height="100%" fill="#f8fafc"/>')
    lines.append(
        '<text x="90" y="32" font-family="Menlo, monospace" font-size="20" fill="#0f172a">TABREWIND 20k Embedding Sweep (workers 0..20)</text>'
    )
    lines.append(
        f'<text x="90" y="52" font-family="Menlo, monospace" font-size="12" fill="#334155">sentences={sentence_count}, batch={batch_size}, slots={slots}, baseline={baseline:.2f}s, best={best_workers} ({best_seconds:.2f}s)</text>'
    )

    for y_px, y_val in h_grid:
        lines.append(
            f'<line x1="{margin_left}" y1="{y_px:.2f}" x2="{margin_left + plot_w}" y2="{y_px:.2f}" stroke="#cbd5e1" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{y_px + 4:.2f}" text-anchor="end" font-family="Menlo, monospace" font-size="11" fill="#475569">{y_val:.2f}s</text>'
        )

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#0f172a" stroke-width="2"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#0f172a" stroke-width="2"/>'
    )

    for tick in x_ticks:
        x = x_to_px(tick)
        lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top + plot_h}" x2="{x:.2f}" y2="{margin_top + plot_h + 6}" stroke="#0f172a" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Menlo, monospace" font-size="11" fill="#334155">{tick}</text>'
        )

    lines.append(
        f'<text x="{margin_left + plot_w / 2:.2f}" y="{height - 55}" text-anchor="middle" font-family="Menlo, monospace" font-size="12" fill="#0f172a">Worker count</text>'
    )
    lines.append(
        f'<text x="22" y="{margin_top + plot_h / 2:.2f}" transform="rotate(-90 22,{margin_top + plot_h / 2:.2f})" text-anchor="middle" font-family="Menlo, monospace" font-size="12" fill="#0f172a">Elapsed seconds (lower is better)</text>'
    )

    lines.append(f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2.5"/>')

    for row in timings:
        if not isinstance(row, dict):
            continue
        workers = int(row["workers"])
        seconds = float(row["seconds"])
        x = x_to_px(workers)
        y = y_to_px(seconds)
        color = "#dc2626" if workers == best_workers else "#2563eb"
        radius = 4 if workers == best_workers else 2.8
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>')

    lines.append(f'<circle cx="{base_x:.2f}" cy="{base_y:.2f}" r="4" fill="#7c3aed"/>')
    lines.append(
        f'<text x="{base_x + 10:.2f}" y="{base_y - 8:.2f}" font-family="Menlo, monospace" font-size="11" fill="#6d28d9">baseline 0w: {baseline:.2f}s</text>'
    )
    lines.append(
        f'<text x="{best_x + 10:.2f}" y="{best_y - 8:.2f}" font-family="Menlo, monospace" font-size="11" fill="#991b1b">best {best_workers}w: {best_seconds:.2f}s ({baseline / best_seconds:.2f}x)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> int:
    output_dir = Path("benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vectorization_sweep_20k.json"
    svg_path = output_dir / "vectorization_sweep_20k.svg"

    data = run_sweep(sentence_count=20_000, max_workers=20, batch_size=32, seed=1337)
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    svg_path.write_text(render_svg(data), encoding="utf-8")

    print(json.dumps({"json_path": str(json_path), "svg_path": str(svg_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
