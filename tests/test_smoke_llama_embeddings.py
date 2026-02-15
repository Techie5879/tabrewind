from __future__ import annotations

import os
import unittest

from main import LlamaCppVectorClient, benchmark_worker_sweep, fetch_server_slots


class LiveLlamaCppVectorSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = os.environ.get("TABREWIND_SMOKE_BASE_URL", "http://127.0.0.1:8080")
        cls.model = os.environ.get("TABREWIND_SMOKE_MODEL", "embeddinggemma")
        cls.api_key = os.environ.get("TABREWIND_SMOKE_API_KEY", "no-key")
        cls.client = LlamaCppVectorClient(
            base_url=cls.base_url,
            model=cls.model,
            timeout_seconds=20.0,
            api_key=cls.api_key,
        )

        try:
            probe = cls.client.encode_text("tabrewind smoke probe")
        except Exception as exc:  # pragma: no cover - live dependency
            raise unittest.SkipTest(f"llama-server not available for smoke tests: {exc}")

        if not probe:
            raise unittest.SkipTest("llama-server returned an empty embedding for probe")

    def test_openai_embeddings_two_inputs(self) -> None:
        vectors = self.client.encode_many(["hello world", "tabrewind smoke test"])
        self.assertEqual(len(vectors), 2)
        self.assertGreater(len(vectors[0]), 8)
        self.assertEqual(len(vectors[0]), len(vectors[1]))

    def test_concurrency_worker_sweep_zero_to_twenty(self) -> None:
        sentences = [
            f"tabrewind live benchmark sentence {index} token {(index * 19) % 29}"
            for index in range(20)
        ]
        result = benchmark_worker_sweep(
            self.client,
            sentences=sentences,
            max_workers=20,
        )

        self.assertEqual(result.sentence_count, 20)
        self.assertEqual(len(result.timings), 21)
        self.assertEqual([timing.workers for timing in result.timings], list(range(21)))
        self.assertGreater(result.vector_dimensions, 8)
        self.assertGreater(result.baseline_seconds, 0.0)
        for timing in result.timings:
            self.assertGreater(timing.elapsed_seconds, 0.0)

        slots = fetch_server_slots(self.base_url, timeout_seconds=5.0)
        if slots is not None and slots > 1:
            faster_parallel_runs = [
                timing
                for timing in result.timings[1:]
                if timing.elapsed_seconds < result.baseline_seconds
            ]
            self.assertTrue(
                faster_parallel_runs,
                msg=(
                    "Expected at least one workers>0 run to beat baseline "
                    "when llama-server has multiple slots"
                ),
            )


if __name__ == "__main__":
    unittest.main()
