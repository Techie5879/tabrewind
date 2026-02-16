from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import main
import sqlite_vec


def _create_places_db(path: Path, rows: list[tuple[int, str, str, int]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE moz_places (id INTEGER PRIMARY KEY, url TEXT, title TEXT)"
    )
    conn.execute(
        "CREATE TABLE moz_historyvisits (id INTEGER PRIMARY KEY, place_id INTEGER, visit_date INTEGER)"
    )

    place_id = 1
    for visit_id, url, title, visit_date in rows:
        conn.execute(
            "INSERT INTO moz_places (id, url, title) VALUES (?, ?, ?)",
            (place_id, url, title),
        )
        conn.execute(
            "INSERT INTO moz_historyvisits (id, place_id, visit_date) VALUES (?, ?, ?)",
            (visit_id, place_id, visit_date),
        )
        place_id += 1

    conn.commit()
    conn.close()


class TestIngestDb(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

        zen_profiles = self.root / "zen_profiles"
        profile = zen_profiles / "profile_a"
        profile.mkdir(parents=True)

        now_us = int(time.time() * 1_000_000)
        one_day_us = 86_400 * 1_000_000
        rows = [
            (1001, "https://example.com/path#frag1", "Example", now_us - 1000),
            (1002, "https://example.com/path#frag2", "Example", now_us - 900),
            (1003, "https://other.test/page", "Other", now_us - 800),
            (1004, "https://old.test/page", "Old", now_us - (3 * one_day_us)),
        ]
        _create_places_db(profile / "places.sqlite", rows)

        self.db_path = self.root / "tabrewind.sqlite"
        self.config = main.AppConfig()
        self.config.profiles.zen_root = str(zen_profiles)
        self.config.profiles.firefox_root = str(self.root / "firefox_profiles")
        self.config.ingest.browsers = ["zen"]
        self.config.ingest.embedding_batch_size = 2

    def _add_profile_rows(self, profile_name: str, rows: list[tuple[int, str, str, int]]) -> None:
        profile_root = Path(self.config.profiles.zen_root)
        profile = profile_root / profile_name
        profile.mkdir(parents=True, exist_ok=True)
        _create_places_db(profile / "places.sqlite", rows)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _fake_encoder(self, model_name: str, texts: list[str]) -> list[list[float]]:
        self.assertTrue(model_name)
        vectors: list[list[float]] = []
        for idx, _ in enumerate(texts):
            vectors.append([float(idx), 0.1, 0.2, 0.3])
        return vectors

    def test_ingest_db_writes_rows_and_search_tables(self) -> None:
        result = main.run_ingest_db(
            self.config,
            db_path=self.db_path,
            since_days=2.0,
            encode_many=self._fake_encoder,
            embedding_dimensions=4,
            embedding_model_name="test-model",
        )

        self.assertEqual(result["discovered_profiles"], 1)
        self.assertEqual(result["source_rows"], 3)
        self.assertEqual(result["pages_touched"], 2)
        self.assertEqual(result["visits_inserted"], 3)
        self.assertEqual(result["fts_rows_synced"], 2)
        self.assertEqual(result["embeddings_synced"], 2)
        self.assertEqual(result["load_errors"], [])
        self.assertEqual(result["embed_errors"], [])

        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        visits = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        fts = conn.execute("SELECT COUNT(*) FROM pages_fts").fetchone()[0]
        emb_meta = conn.execute("SELECT COUNT(*) FROM page_embeddings").fetchone()[0]
        emb_vec = conn.execute("SELECT COUNT(*) FROM page_embedding_vec").fetchone()[0]
        active_models = conn.execute(
            "SELECT COUNT(*) FROM embedding_models WHERE is_active = 1"
        ).fetchone()[0]
        profile_key = conn.execute("SELECT profile_key FROM source_profiles").fetchone()[0]
        conn.close()

        self.assertEqual(pages, 2)
        self.assertEqual(visits, 3)
        self.assertEqual(fts, 2)
        self.assertEqual(emb_meta, 2)
        self.assertEqual(emb_vec, 2)
        self.assertEqual(active_models, 1)
        self.assertEqual(len(profile_key), 64)

    def test_ingest_db_is_idempotent(self) -> None:
        first = main.run_ingest_db(
            self.config,
            db_path=self.db_path,
            since_days=2.0,
            encode_many=self._fake_encoder,
            embedding_dimensions=4,
            embedding_model_name="test-model",
        )
        second = main.run_ingest_db(
            self.config,
            db_path=self.db_path,
            since_days=2.0,
            encode_many=self._fake_encoder,
            embedding_dimensions=4,
            embedding_model_name="test-model",
        )

        self.assertEqual(first["visits_inserted"], 3)
        self.assertEqual(second["visits_inserted"], 0)

        conn = sqlite3.connect(self.db_path)
        pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        visits = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        conn.close()
        self.assertEqual(pages, 2)
        self.assertEqual(visits, 3)

    def test_ingest_db_embeds_all_touched_pages(self) -> None:
        result = main.run_ingest_db(
            self.config,
            db_path=self.db_path,
            since_days=2.0,
            encode_many=self._fake_encoder,
            embedding_dimensions=4,
            embedding_model_name="test-model",
        )

        self.assertEqual(result["pages_touched"], 2)
        self.assertEqual(result["embeddings_synced"], 2)

    def test_embedding_fallback_recovers_from_batch_failures(self) -> None:
        stats = {"calls": 0, "batch_failures": 0}

        def flaky_encoder(model_name: str, texts: list[str]) -> list[list[float]]:
            self.assertTrue(model_name)
            stats["calls"] += 1
            if any("\n" in text for text in texts):
                stats["batch_failures"] += 1
                raise RuntimeError("simulated token overflow")
            return [[0.11, 0.22, 0.33, 0.44] for _ in texts]

        result = main.run_ingest_db(
            self.config,
            db_path=self.db_path,
            since_days=2.0,
            encode_many=flaky_encoder,
            embedding_dimensions=4,
            embedding_model_name="test-model",
        )

        self.assertGreater(stats["calls"], 0)
        self.assertGreater(stats["batch_failures"], 0)
        self.assertEqual(result["pages_touched"], 2)
        self.assertEqual(result["embeddings_synced"], 2)
        self.assertEqual(result["embed_errors"], [])

    def test_ingest_db_applies_domain_rules_with_last_match_wins(self) -> None:
        self.config.ingest.domain_rules = [
            "-example.com",
            "+example.com",
            "-example.com",
        ]

        result = main.run_ingest_db(
            self.config,
            db_path=self.db_path,
            since_days=2.0,
            encode_many=self._fake_encoder,
            embedding_dimensions=4,
            embedding_model_name="test-model",
        )

        self.assertEqual(result["source_rows"], 3)
        self.assertEqual(result["pages_touched"], 1)
        self.assertEqual(result["visits_inserted"], 1)
        self.assertEqual(result["embeddings_synced"], 1)
        self.assertEqual(result["filtered_denied_rows"], 2)

    def test_ingest_db_default_allow_when_no_rule_matches(self) -> None:
        self.config.ingest.domain_rules = ["-blocked.example"]

        result = main.run_ingest_db(
            self.config,
            db_path=self.db_path,
            since_days=2.0,
            encode_many=self._fake_encoder,
            embedding_dimensions=4,
            embedding_model_name="test-model",
        )

        self.assertEqual(result["source_rows"], 3)
        self.assertEqual(result["pages_touched"], 2)
        self.assertEqual(result["visits_inserted"], 3)
        self.assertEqual(result["filtered_denied_rows"], 0)

    def test_ingest_db_multi_domain_rules_cover_subdomain_and_url_path_inputs(self) -> None:
        now_us = int(time.time() * 1_000_000)
        self._add_profile_rows(
            "profile_b",
            [
                (2001, "https://example.com/alpha", "Example Root", now_us - 700),
                (2002, "https://sub.example.com/alpha", "Sub Example", now_us - 690),
                (2003, "https://allow.example.com/path?a=1#section", "Allow Example", now_us - 680),
                (2004, "https://secure.example.com/private/banking", "Secure Example", now_us - 670),
                (2005, "https://other.test/path/segment", "Other Domain", now_us - 660),
            ],
        )

        self.config.ingest.domain_rules = [
            "-*.example.com",
            "+allow.example.com",
            "-secure.example.com/private",  # normalized to host-only pattern
            "-other.test",
        ]

        result = main.run_ingest_db(
            self.config,
            db_path=self.db_path,
            since_days=2.0,
            encode_many=self._fake_encoder,
            embedding_dimensions=4,
            embedding_model_name="test-model",
        )

        self.assertEqual(result["source_rows"], 8)
        self.assertEqual(result["filtered_allowed_rows"], 4)
        self.assertEqual(result["filtered_denied_rows"], 4)
        self.assertEqual(result["visits_inserted"], 4)
        self.assertEqual(result["pages_touched"], 3)
        self.assertEqual(result["embeddings_synced"], 3)

        conn = sqlite3.connect(self.db_path)
        hosts = {row[0] for row in conn.execute("SELECT host FROM pages ORDER BY host")}
        conn.close()
        self.assertEqual(hosts, {"allow.example.com", "example.com"})

    def test_ingest_db_path_rule_blocks_only_matching_paths(self) -> None:
        now_us = int(time.time() * 1_000_000)
        self._add_profile_rows(
            "profile_path",
            [
                (3001, "https://pathonly.test/private/a", "Private", now_us - 500),
                (3002, "https://pathonly.test/public/a", "Public", now_us - 490),
            ],
        )
        self.config.ingest.domain_rules = ["-pathonly.test/private/*"]

        result = main.run_ingest_db(
            self.config,
            db_path=self.db_path,
            since_days=2.0,
            encode_many=self._fake_encoder,
            embedding_dimensions=4,
            embedding_model_name="test-model",
        )

        self.assertEqual(result["source_rows"], 5)
        self.assertEqual(result["filtered_denied_rows"], 1)
        self.assertEqual(result["filtered_allowed_rows"], 4)
        self.assertEqual(result["visits_inserted"], 4)
        self.assertEqual(result["pages_touched"], 3)
        self.assertEqual(result["embeddings_synced"], 3)

        conn = sqlite3.connect(self.db_path)
        urls = {row[0] for row in conn.execute("SELECT url FROM pages ORDER BY url")}
        conn.close()
        self.assertIn("https://pathonly.test/public/a", urls)
        self.assertNotIn("https://pathonly.test/private/a", urls)
