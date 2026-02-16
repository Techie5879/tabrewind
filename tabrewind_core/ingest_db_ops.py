from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

import schema
import sqlite_vec

from .app_types import AppConfig, ProfileStore
from .config_store import SUPPORTED_BROWSERS
from .domain_policy import compile_domain_rules, resolve_domain_policy
from .history_ops import discover_places_files, normalize_url
from .vectorization import VectorizationClient


def _stable_profile_key(store: ProfileStore) -> str:
    material = f"{store.browser}:{store.db_path.resolve()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _resolve_source_profile_id(conn: sqlite3.Connection, store: ProfileStore) -> int:
    profile_key = _stable_profile_key(store)
    row = conn.execute(
        "SELECT source_profile_id FROM source_profiles WHERE profile_key = ?",
        (profile_key,),
    ).fetchone()
    if row is not None:
        return int(row[0])

    conn.execute(
        "INSERT INTO source_profiles (browser, profile_key, label) VALUES (?, ?, NULL)",
        (store.browser, profile_key),
    )
    created = conn.execute(
        "SELECT source_profile_id FROM source_profiles WHERE profile_key = ?",
        (profile_key,),
    ).fetchone()
    if created is None:
        raise RuntimeError("Failed to resolve source profile id after insert")
    return int(created[0])


def _load_recent_visit_rows(
    store: ProfileStore,
    cutoff_us: int,
) -> tuple[list[tuple[int, str, str, int]], str | None]:
    uri = f"file:{store.db_path.as_posix()}?mode=ro&immutable=1"
    query = (
        "SELECT v.id, p.url, COALESCE(p.title, ''), COALESCE(v.visit_date, 0) "
        "FROM moz_historyvisits AS v "
        "JOIN moz_places AS p ON p.id = v.place_id "
        "WHERE p.url IS NOT NULL AND TRIM(p.url) != '' "
        "AND COALESCE(v.visit_date, 0) >= ? "
        "ORDER BY COALESCE(v.visit_date, 0) DESC, v.id DESC"
    )

    profile_name = store.db_path.parent.name
    try:
        with sqlite3.connect(uri, uri=True, timeout=1.0) as conn:
            cursor = conn.execute(query, (cutoff_us,))
            rows = [
                (int(source_visit_id), str(raw_url), str(raw_title), int(visited_at_us))
                for source_visit_id, raw_url, raw_title, visited_at_us in cursor
            ]
    except sqlite3.Error as exc:
        return [], f"{store.browser}:{profile_name} ({store.db_path.name}) -> {exc}"

    return rows, None


def _upsert_page_row(
    conn: sqlite3.Connection,
    *,
    canonical_url: str,
    host: str,
    title: str,
    observed_at_us: int,
) -> int:
    conn.execute(
        "INSERT INTO pages (url, host, title, first_seen_us, last_seen_us, normalization_version) "
        "VALUES (?, ?, ?, ?, ?, 1) "
        "ON CONFLICT(url) DO UPDATE SET "
        "host = excluded.host, "
        "title = CASE WHEN excluded.title != '' THEN excluded.title ELSE pages.title END, "
        "first_seen_us = MIN(pages.first_seen_us, excluded.first_seen_us), "
        "last_seen_us = MAX(pages.last_seen_us, excluded.last_seen_us)",
        (canonical_url, host, title, observed_at_us, observed_at_us),
    )
    row = conn.execute("SELECT page_id FROM pages WHERE url = ?", (canonical_url,)).fetchone()
    if row is None:
        raise RuntimeError("Failed to resolve page_id after upsert")
    return int(row[0])


def _sync_pages_fts(conn: sqlite3.Connection, page_ids: list[int]) -> int:
    synced = 0
    for page_id in page_ids:
        row = conn.execute(
            "SELECT title, url FROM pages WHERE page_id = ?",
            (page_id,),
        ).fetchone()
        if row is None:
            continue
        title, url = row
        bookmark_row = conn.execute(
            "SELECT GROUP_CONCAT(TRIM(COALESCE(title_override, '') || ' ' || COALESCE(notes, '')), ' ') "
            "FROM bookmarks WHERE page_id = ? AND deleted_at_us IS NULL",
            (page_id,),
        ).fetchone()
        bookmark_text = ""
        if bookmark_row is not None and isinstance(bookmark_row[0], str):
            bookmark_text = bookmark_row[0]

        conn.execute(
            "INSERT OR REPLACE INTO pages_fts(rowid, title, url, bookmark_text) VALUES (?, ?, ?, ?)",
            (page_id, title, url, bookmark_text),
        )
        synced += 1
    return synced


def _resolve_active_embedding_model(conn: sqlite3.Connection) -> tuple[int, str, int]:
    rows = conn.execute(
        "SELECT model_id, model_name, dimensions FROM embedding_models WHERE is_active = 1"
    ).fetchall()
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one active embedding model")
    model_id, model_name, dimensions = rows[0]
    return int(model_id), str(model_name), int(dimensions)


def _embedding_text_candidates(title: str, url: str) -> list[str]:
    normalized_title = title.strip()
    normalized_url = url.strip()
    parsed = urlsplit(normalized_url)
    url_without_query = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))

    candidates = [
        f"title: {normalized_title}\nurl: {normalized_url}",
        f"title: {normalized_title[:512]}\nurl: {normalized_url[:2048]}",
        f"title: {normalized_title[:256]}\nurl: {url_without_query[:1024]}",
        f"title: {normalized_title[:128]}\nurl: {url_without_query[:512]}",
        f"url: {url_without_query[:512]}",
        f"url: {normalized_url[:512]}",
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _encode_one_embedding_with_fallback(
    *,
    model_name: str,
    title: str,
    url: str,
    dimensions: int,
    encode_many: Callable[[str, list[str]], list[list[float]]],
) -> tuple[list[float] | None, str | None]:
    last_error = "embedding fallback exhausted"
    for attempt_index, text in enumerate(_embedding_text_candidates(title, url), start=1):
        try:
            vectors = encode_many(model_name, [text])
        except Exception as exc:
            last_error = f"fallback attempt {attempt_index} failed: {exc}"
            continue
        if len(vectors) != 1:
            last_error = (
                f"fallback attempt {attempt_index} length mismatch: "
                f"got {len(vectors)} expected 1"
            )
            continue

        vector = vectors[0]
        if len(vector) != dimensions:
            last_error = (
                f"fallback attempt {attempt_index} dimension mismatch: "
                f"got {len(vector)} expected {dimensions}"
            )
            continue

        return vector, None

    return None, last_error


def _sync_page_embeddings(
    conn: sqlite3.Connection,
    *,
    page_ids: list[int],
    model_id: int,
    model_name: str,
    dimensions: int,
    batch_size: int,
    encode_many: Callable[[str, list[str]], list[list[float]]],
) -> tuple[int, list[str]]:
    page_rows_with_recency: list[tuple[int, str, str, int]] = []
    for page_id in sorted(set(page_ids)):
        row = conn.execute(
            "SELECT page_id, title, url, last_seen_us FROM pages WHERE page_id = ?",
            (page_id,),
        ).fetchone()
        if row is None:
            continue
        page_rows_with_recency.append((int(row[0]), str(row[1]), str(row[2]), int(row[3])))

    page_rows_with_recency.sort(key=lambda row: (-row[3], row[0]))
    page_rows = [(page_id, title, url) for page_id, title, url, _ in page_rows_with_recency]

    synced = 0
    errors: list[str] = []
    resolved_batch_size = max(1, batch_size)

    def persist_vector(page_id: int, title: str, url: str, vector: list[float]) -> None:
        nonlocal synced
        now_us = int(time.time() * 1_000_000)
        content_hash = hashlib.sha256(f"{title}\n{url}".encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT INTO page_embeddings (page_id, model_id, content_hash, updated_at_us) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(page_id) DO UPDATE SET "
            "model_id = excluded.model_id, "
            "content_hash = excluded.content_hash, "
            "updated_at_us = excluded.updated_at_us",
            (page_id, model_id, content_hash, now_us),
        )
        conn.execute(
            "DELETE FROM page_embedding_vec WHERE rowid = ?",
            (page_id,),
        )
        conn.execute(
            "INSERT INTO page_embedding_vec(rowid, embedding) VALUES (?, ?)",
            (page_id, sqlite_vec.serialize_float32(vector)),
        )
        synced += 1

    def encode_single_with_fallback(page_id: int, title: str, url: str, reason: str) -> None:
        vector, fallback_error = _encode_one_embedding_with_fallback(
            model_name=model_name,
            title=title,
            url=url,
            dimensions=dimensions,
            encode_many=encode_many,
        )
        if vector is None:
            errors.append(f"page_id={page_id} {reason}; {fallback_error}")
            return
        persist_vector(page_id, title, url, vector)

    for start in range(0, len(page_rows), resolved_batch_size):
        batch = page_rows[start : start + resolved_batch_size]
        texts = [f"title: {title}\nurl: {url}" for _, title, url in batch]
        try:
            vectors = encode_many(model_name, texts)
        except Exception as exc:
            for page_id, title, url in batch:
                encode_single_with_fallback(
                    page_id,
                    title,
                    url,
                    reason=f"batch offset {start} failed: {exc}",
                )
            continue

        if len(vectors) != len(batch):
            for page_id, title, url in batch:
                encode_single_with_fallback(
                    page_id,
                    title,
                    url,
                    reason=(
                        f"batch length mismatch at offset {start}: "
                        f"got {len(vectors)} expected {len(batch)}"
                    ),
                )
            continue

        for (page_id, title, url), vector in zip(batch, vectors, strict=True):
            if len(vector) != dimensions:
                encode_single_with_fallback(
                    page_id,
                    title,
                    url,
                    reason=(
                        f"batch dimension mismatch for page_id={page_id}: "
                        f"got {len(vector)} expected {dimensions}"
                    ),
                )
                continue
            persist_vector(page_id, title, url, vector)

    return synced, errors


def run_ingest_db(
    config: AppConfig,
    *,
    db_path: Path,
    since_days: float,
    browsers_override: list[str] | None = None,
    encode_many: Callable[[str, list[str]], list[list[float]]] | None = None,
    embedding_dimensions: int | None = None,
    embedding_model_name: str | None = None,
) -> dict[str, object]:
    compiled_domain_rules = compile_domain_rules(config.ingest.domain_rules)
    selected_browsers = _resolve_selected_browsers(config, browsers_override)

    discovered_profiles: list[ProfileStore] = []
    for browser in selected_browsers:
        root = config.profiles.zen_root if browser == "zen" else config.profiles.firefox_root
        discovered_profiles.extend(discover_places_files(browser, root))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    cutoff_us = int((time.time() - max(0.0, since_days) * 86_400.0) * 1_000_000)
    load_errors: list[str] = []
    source_rows = 0
    visits_inserted = 0
    touched_pages: set[int] = set()
    filtered_denied_rows = 0
    filtered_allowed_rows = 0

    with sqlite3.connect(db_path) as conn:
        schema.init_schema(
            conn,
            embedding_dimensions=embedding_dimensions,
            embedding_model_name=(embedding_model_name or config.llama.model),
            embedding_base_url=config.llama.base_url,
            embedding_timeout_seconds=config.llama.request_timeout_seconds,
            embedding_api_key=config.llama.api_key,
        )

        for store in discovered_profiles:
            source_profile_id = _resolve_source_profile_id(conn, store)
            rows, error = _load_recent_visit_rows(
                store,
                cutoff_us=cutoff_us,
            )
            if error is not None:
                load_errors.append(error)
                continue

            source_rows += len(rows)
            for source_visit_id, raw_url, raw_title, visited_at_us in rows:
                canonical = normalize_url(raw_url)
                if canonical is None:
                    continue
                host = (urlsplit(canonical).hostname or "").lower()
                if not host:
                    continue
                decision = resolve_domain_policy(canonical, compiled_domain_rules)
                if not decision.allowed:
                    filtered_denied_rows += 1
                    continue
                filtered_allowed_rows += 1

                title = raw_title.strip() if isinstance(raw_title, str) else ""
                if not title:
                    title = "(untitled)"

                page_id = _upsert_page_row(
                    conn,
                    canonical_url=canonical,
                    host=host,
                    title=title,
                    observed_at_us=visited_at_us,
                )
                touched_pages.add(page_id)

                cursor = conn.execute(
                    "INSERT OR IGNORE INTO visits (page_id, visited_at_us, title_snapshot, source_profile_id, source_visit_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (page_id, visited_at_us, title, source_profile_id, source_visit_id),
                )
                if cursor.rowcount == 1:
                    visits_inserted += 1

        fts_synced = _sync_pages_fts(conn, sorted(touched_pages))
        model_id, model_name, dimensions = _resolve_active_embedding_model(conn)

        def _default_encode_many(active_model_name: str, texts: list[str]) -> list[list[float]]:
            client = VectorizationClient(
                base_url=config.llama.base_url,
                model=active_model_name,
                timeout_seconds=config.llama.request_timeout_seconds,
                api_key=config.llama.api_key,
            )
            return client.encode_many(texts)

        encoder = _default_encode_many if encode_many is None else encode_many
        embeddings_synced, embed_errors = _sync_page_embeddings(
            conn,
            page_ids=sorted(touched_pages),
            model_id=model_id,
            model_name=model_name,
            dimensions=dimensions,
            batch_size=config.ingest.embedding_batch_size,
            encode_many=encoder,
        )

    return {
        "db_path": str(db_path),
        "discovered_profiles": len(discovered_profiles),
        "source_rows": source_rows,
        "pages_touched": len(touched_pages),
        "visits_inserted": visits_inserted,
        "filtered_allowed_rows": filtered_allowed_rows,
        "filtered_denied_rows": filtered_denied_rows,
        "fts_rows_synced": fts_synced,
        "embeddings_synced": embeddings_synced,
        "load_errors": load_errors,
        "embed_errors": embed_errors,
    }


def _resolve_selected_browsers(config: AppConfig, browsers_override: list[str] | None) -> list[str]:
    selected_browsers = (
        [browser for browser in browsers_override if browser in SUPPORTED_BROWSERS]
        if browsers_override is not None
        else [browser for browser in config.ingest.browsers if browser in SUPPORTED_BROWSERS]
    )
    if not selected_browsers:
        selected_browsers = ["zen", "firefox"]
    return selected_browsers


def collect_recent_hosts(
    config: AppConfig,
    *,
    since_days: float,
    browsers_override: list[str] | None,
) -> tuple[list[str], list[str]]:
    selected_browsers = _resolve_selected_browsers(config, browsers_override)
    discovered_profiles: list[ProfileStore] = []
    for browser in selected_browsers:
        root = config.profiles.zen_root if browser == "zen" else config.profiles.firefox_root
        discovered_profiles.extend(discover_places_files(browser, root))

    cutoff_us = int((time.time() - max(0.0, since_days) * 86_400.0) * 1_000_000)
    hosts: set[str] = set()
    errors: list[str] = []
    for store in discovered_profiles:
        rows, error = _load_recent_visit_rows(store, cutoff_us=cutoff_us)
        if error is not None:
            errors.append(error)
            continue
        for _, raw_url, _, _ in rows:
            canonical = normalize_url(raw_url)
            if canonical is None:
                continue
            host = (urlsplit(canonical).hostname or "").lower()
            if host:
                hosts.add(host)

    return sorted(hosts), errors


__all__ = ["collect_recent_hosts", "run_ingest_db"]
