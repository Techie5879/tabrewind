"""Unittest suite for TABREWIND v1 schema init and invariants."""

from __future__ import annotations

import importlib.util
import re
import sqlite3
import sys
import unittest
from pathlib import Path


def _schema_module():
    mod_path = Path(__file__).resolve().parents[1] / "schema.py"
    spec = importlib.util.spec_from_file_location("tabrewind_schema", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load schema module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class SchemaTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = _schema_module()
        self.conn = sqlite3.connect(":memory:")

    def tearDown(self) -> None:
        self.conn.close()

    def init_schema_local(self, *, dimensions: int = 8, model_name: str = "test_embedding") -> None:
        self.schema.init_schema(
            self.conn,
            embedding_dimensions=dimensions,
            embedding_model_name=model_name,
        )

    def table_exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def table_columns(self, table: str) -> list[str]:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [row[1] for row in rows]

    def index_names(self, table: str) -> list[str]:
        rows = self.conn.execute(f"PRAGMA index_list({table})").fetchall()
        return [row[1] for row in rows]

    def index_key_columns(self, index_name: str) -> list[tuple[str, bool]]:
        rows = self.conn.execute(f"PRAGMA index_xinfo({index_name})").fetchall()
        return [(row[2], bool(row[3])) for row in rows if row[5] == 1]

    def foreign_keys(self, table: str) -> list[tuple[str, str, str]]:
        rows = self.conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        return [(row[2], row[3], row[4]) for row in rows]


class TestForeignKeys(SchemaTestCase):
    def test_foreign_keys_enabled_after_init(self) -> None:
        self.init_schema_local()
        row = self.conn.execute("PRAGMA foreign_keys").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)


class TestCoreTables(SchemaTestCase):
    def test_schema_creates_core_tables_and_columns(self) -> None:
        self.init_schema_local()

        expected = {
            "source_profiles": ["source_profile_id", "browser", "profile_key", "label"],
            "pages": [
                "page_id",
                "url",
                "host",
                "title",
                "first_seen_us",
                "last_seen_us",
                "normalization_version",
            ],
            "visits": [
                "visit_id",
                "page_id",
                "visited_at_us",
                "title_snapshot",
                "source_profile_id",
                "source_visit_id",
            ],
            "bookmark_folders": [
                "folder_id",
                "parent_folder_id",
                "title",
                "position",
                "created_at_us",
                "updated_at_us",
            ],
            "bookmarks": [
                "bookmark_id",
                "page_id",
                "folder_id",
                "title_override",
                "notes",
                "status",
                "deleted_at_us",
                "created_at_us",
                "updated_at_us",
            ],
        }

        for table, columns in expected.items():
            self.assertTrue(self.table_exists(table), f"missing table {table}")
            self.assertEqual(self.table_columns(table), columns)


class TestVisitsInvariants(SchemaTestCase):
    def test_source_profile_fk_and_not_null(self) -> None:
        self.init_schema_local()
        self.conn.execute(
            "INSERT INTO source_profiles (source_profile_id, browser, profile_key) VALUES (1, 'zen', 'pk1')"
        )
        self.conn.execute(
            "INSERT INTO pages (page_id, url, host, title, first_seen_us, last_seen_us, normalization_version) "
            "VALUES (1, 'https://example.com/', 'example.com', 'Example', 0, 0, 1)"
        )
        self.conn.execute(
            "INSERT INTO visits (visit_id, page_id, visited_at_us, title_snapshot, source_profile_id, source_visit_id) "
            "VALUES (1, 1, 0, 'Example', 1, 100)"
        )

        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO visits (visit_id, page_id, visited_at_us, title_snapshot, source_profile_id, source_visit_id) "
                "VALUES (2, 1, 1, 'x', NULL, 101)"
            )

        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO visits (visit_id, page_id, visited_at_us, title_snapshot, source_profile_id, source_visit_id) "
                "VALUES (3, 1, 1, 'x', 999, 101)"
            )

    def test_visit_idempotency_unique_constraint(self) -> None:
        self.init_schema_local()
        self.conn.execute(
            "INSERT INTO source_profiles (source_profile_id, browser, profile_key) VALUES (1, 'zen', 'pk1')"
        )
        self.conn.execute(
            "INSERT INTO pages (page_id, url, host, title, first_seen_us, last_seen_us, normalization_version) "
            "VALUES (1, 'https://example.com/', 'example.com', 'Example', 0, 0, 1)"
        )
        self.conn.execute(
            "INSERT INTO visits (visit_id, page_id, visited_at_us, title_snapshot, source_profile_id, source_visit_id) "
            "VALUES (1, 1, 0, 'Example', 1, 100)"
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO visits (visit_id, page_id, visited_at_us, title_snapshot, source_profile_id, source_visit_id) "
                "VALUES (2, 1, 1, 'Other', 1, 100)"
            )


class TestIndexDefinitions(SchemaTestCase):
    def test_expected_indexes_exist_with_exact_columns_and_sort_order(self) -> None:
        self.init_schema_local()

        expected = {
            "idx_pages_last_seen": [(
                "last_seen_us",
                True,
            )],
            "idx_pages_host_last_seen": [("host", False), ("last_seen_us", True)],
            "idx_visits_page_time": [("page_id", False), ("visited_at_us", True)],
            "idx_visits_time": [("visited_at_us", True)],
            "idx_bookmarks_status_updated": [("status", False), ("updated_at_us", True)],
            "idx_bookmarks_folder_updated": [("folder_id", False), ("updated_at_us", True)],
        }

        by_table = {
            "pages": ["idx_pages_last_seen", "idx_pages_host_last_seen"],
            "visits": ["idx_visits_page_time", "idx_visits_time"],
            "bookmarks": ["idx_bookmarks_status_updated", "idx_bookmarks_folder_updated"],
        }

        for table, names in by_table.items():
            existing = set(self.index_names(table))
            for name in names:
                self.assertIn(name, existing, f"missing index {name}")
                self.assertEqual(self.index_key_columns(name), expected[name])


class TestBookmarksDeleteField(SchemaTestCase):
    def test_bookmarks_uses_deleted_at_us_not_is_deleted(self) -> None:
        self.init_schema_local()
        columns = self.table_columns("bookmarks")
        self.assertIn("deleted_at_us", columns)
        self.assertNotIn("is_deleted", columns)


class TestFtsRowidPolicy(SchemaTestCase):
    def test_pages_fts_exists(self) -> None:
        self.init_schema_local()
        self.assertTrue(self.table_exists("pages_fts"))

    def test_fts_insert_select_and_delete_by_rowid(self) -> None:
        self.init_schema_local()
        self.conn.execute(
            "INSERT INTO pages (page_id, url, host, title, first_seen_us, last_seen_us, normalization_version) "
            "VALUES (42, 'https://a.com/', 'a.com', 'Page A', 0, 0, 1)"
        )
        self.conn.execute(
            "INSERT INTO pages_fts(rowid, title, url, bookmark_text) VALUES (42, 'Page A', 'https://a.com/', '')"
        )

        row = self.conn.execute("SELECT rowid FROM pages_fts WHERE rowid = 42").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 42)

        self.conn.execute("INSERT INTO pages_fts(pages_fts, rowid) VALUES('delete', 42)")
        row_after = self.conn.execute("SELECT rowid FROM pages_fts WHERE rowid = 42").fetchone()
        self.assertIsNone(row_after)


class TestEmbeddingModels(SchemaTestCase):
    def test_init_inserts_one_active_model(self) -> None:
        self.init_schema_local(dimensions=16, model_name="m_init")
        rows = self.conn.execute(
            "SELECT model_name, dimensions, is_active FROM embedding_models ORDER BY model_id"
        ).fetchall()
        self.assertEqual(rows, [("m_init", 16, 1)])

    def test_two_active_models_are_rejected(self) -> None:
        self.init_schema_local(model_name="m_active")
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO embedding_models (model_name, dimensions, is_active) VALUES ('m2', 8, 1)"
            )

    def test_cannot_transition_to_zero_active_models(self) -> None:
        self.init_schema_local(model_name="m_active")
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute("UPDATE embedding_models SET is_active = 0 WHERE model_name = 'm_active'")

    def test_inactive_model_is_allowed_when_one_active_exists(self) -> None:
        self.init_schema_local(model_name="m_active")
        self.conn.execute(
            "INSERT INTO embedding_models (model_name, dimensions, is_active) VALUES ('m_inactive', 8, 0)"
        )
        total = self.conn.execute("SELECT COUNT(*) FROM embedding_models").fetchone()[0]
        active = self.conn.execute("SELECT COUNT(*) FROM embedding_models WHERE is_active = 1").fetchone()[0]
        self.assertEqual(total, 2)
        self.assertEqual(active, 1)


class TestPageEmbeddings(SchemaTestCase):
    def test_table_has_expected_foreign_keys(self) -> None:
        self.init_schema_local()
        fks = set(self.foreign_keys("page_embeddings"))
        self.assertIn(("pages", "page_id", "page_id"), fks)
        self.assertIn(("embedding_models", "model_id", "model_id"), fks)

    def test_fk_enforcement_for_page_and_model_ids(self) -> None:
        self.init_schema_local(model_name="m1")
        self.conn.execute(
            "INSERT INTO pages (page_id, url, host, title, first_seen_us, last_seen_us, normalization_version) "
            "VALUES (1, 'https://x.com/', 'x.com', 'X', 0, 0, 1)"
        )
        self.conn.execute(
            "INSERT INTO pages (page_id, url, host, title, first_seen_us, last_seen_us, normalization_version) "
            "VALUES (2, 'https://y.com/', 'y.com', 'Y', 0, 0, 1)"
        )

        model_id = self.conn.execute(
            "SELECT model_id FROM embedding_models WHERE model_name = 'm1'"
        ).fetchone()[0]

        self.conn.execute(
            "INSERT INTO page_embeddings (page_id, model_id, content_hash, updated_at_us) "
            "VALUES (1, ?, 'h1', 0)",
            (model_id,),
        )

        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO page_embeddings (page_id, model_id, content_hash, updated_at_us) "
                "VALUES (999, ?, 'h2', 0)",
                (model_id,),
            )

        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO page_embeddings (page_id, model_id, content_hash, updated_at_us) "
                "VALUES (2, 999, 'h3', 0)"
            )


class TestVecTable(SchemaTestCase):
    def test_vec_table_created_with_configured_dimension(self) -> None:
        self.init_schema_local(dimensions=16)
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='page_embedding_vec'"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertIsInstance(row[0], str)
        match = re.search(r"embedding\s+float\[(\d+)\]", row[0], flags=re.IGNORECASE)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 16)

    def test_vec_table_accepts_insert_by_page_rowid(self) -> None:
        import sqlite_vec

        self.init_schema_local(dimensions=8)
        self.conn.execute(
            "INSERT INTO pages (page_id, url, host, title, first_seen_us, last_seen_us, normalization_version) "
            "VALUES (7, 'https://v.com/', 'v.com', 'V', 0, 0, 1)"
        )
        blob = sqlite_vec.serialize_float32([0.1] * 8)
        self.conn.execute(
            "INSERT INTO page_embedding_vec(rowid, embedding) VALUES (7, ?)",
            (blob,),
        )
        row = self.conn.execute("SELECT rowid FROM page_embedding_vec WHERE rowid = 7").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 7)

    def test_vec_table_dimension_mismatch_raises(self) -> None:
        self.init_schema_local(dimensions=8)
        with self.assertRaises(RuntimeError):
            self.schema.init_schema(
                self.conn,
                embedding_dimensions=16,
                embedding_model_name="test_embedding",
            )


class TestLlamaServerIntegration(SchemaTestCase):
    def test_fetch_server_model_info_has_model_and_dimensions(self) -> None:
        model_name, dimensions = self.schema.fetch_server_model_info()
        self.assertIsNotNone(model_name)
        self.assertIsInstance(model_name, str)
        self.assertNotEqual(model_name.strip(), "")
        if dimensions is None:
            dimensions = self.schema.discover_embedding_dimensions(model_name=None)
        self.assertGreater(dimensions, 0)

    def test_discover_embedding_dimensions_calls_embedding_endpoint(self) -> None:
        dimensions = self.schema.discover_embedding_dimensions(model_name=None)
        self.assertGreater(dimensions, 0)

    def test_init_schema_can_autodiscover_model_and_dimensions(self) -> None:
        self.schema.init_schema(
            self.conn,
            embedding_model_name=None,
            embedding_dimensions=None,
        )

        active = self.conn.execute(
            "SELECT model_name, dimensions FROM embedding_models WHERE is_active = 1"
        ).fetchone()
        self.assertIsNotNone(active)
        self.assertIsInstance(active[0], str)
        self.assertGreater(active[1], 0)

        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='page_embedding_vec'"
        ).fetchone()
        self.assertIsNotNone(row)
        match = re.search(r"embedding\s+float\[(\d+)\]", row[0], flags=re.IGNORECASE)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), active[1])
