"""TABREWIND v1 SQLite schema init."""

from __future__ import annotations

import json
import re
import sqlite3
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import sqlite_vec


DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_EMBEDDING_MODEL_NAME = "embeddinggemma"
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 30.0
DEFAULT_EMBEDDING_PROBE_TEXT = "tabrewind embedding dimension probe"
_VEC_DIMENSION_PATTERN = re.compile(r"embedding\s+float\[(\d+)\]", re.IGNORECASE)


def _http_get_json(
    *,
    url: str,
    timeout_seconds: float,
    api_key: str | None,
) -> dict[str, object]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url=url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to connect to llama-server: {exc.reason}") from exc

    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"GET {url} returned unexpected JSON response shape")
    return parsed


def _http_post_json(
    *,
    url: str,
    payload: dict[str, object],
    timeout_seconds: float,
    api_key: str | None,
) -> dict[str, object]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to connect to llama-server: {exc.reason}") from exc

    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"POST {url} returned unexpected JSON response shape")
    return parsed


def _sanitize_model_identifier(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("model identifier must be non-empty")
    normalized = normalized.rsplit("/", maxsplit=1)[-1]
    normalized = normalized.rsplit("\\", maxsplit=1)[-1]
    if not normalized:
        raise ValueError("model identifier must be non-empty")
    return normalized


def fetch_server_model_info(
    *,
    base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    api_key: str | None = None,
) -> tuple[str | None, int | None]:
    """Fetch active model id and embedding size from llama-server metadata endpoints."""
    models_url = f"{base_url.rstrip('/')}/v1/models"
    model_name: str | None = None
    dimensions: int | None = None

    try:
        payload = _http_get_json(url=models_url, timeout_seconds=timeout_seconds, api_key=api_key)
    except RuntimeError:
        payload = {}

    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            identifier = first.get("id")
            if isinstance(identifier, str) and identifier.strip():
                model_name = _sanitize_model_identifier(identifier)
            meta = first.get("meta")
            if isinstance(meta, dict):
                n_embd = meta.get("n_embd")
                if isinstance(n_embd, int) and n_embd > 0:
                    dimensions = n_embd

    if model_name is not None:
        return model_name, dimensions

    props_url = f"{base_url.rstrip('/')}/props"
    props = _http_get_json(url=props_url, timeout_seconds=timeout_seconds, api_key=api_key)
    model_path = props.get("model_path")
    if isinstance(model_path, str) and model_path.strip():
        model_name = _sanitize_model_identifier(model_path)
    return model_name, dimensions


def discover_embedding_dimensions(
    *,
    base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    model_name: str | None = DEFAULT_EMBEDDING_MODEL_NAME,
    timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    api_key: str | None = None,
    probe_text: str = DEFAULT_EMBEDDING_PROBE_TEXT,
) -> int:
    """Discover embedding dimensions by requesting one embedding from llama-server."""
    if not probe_text.strip():
        raise ValueError("probe_text must be non-empty")

    payload: dict[str, object] = {
        "input": [probe_text],
        "encoding_format": "float",
    }
    if model_name is not None and model_name.strip():
        payload["model"] = model_name
    parsed = _http_post_json(
        url=f"{base_url.rstrip('/')}/v1/embeddings",
        payload=payload,
        timeout_seconds=timeout_seconds,
        api_key=api_key,
    )

    data = parsed.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError("llama-server response missing embeddings data")

    first = data[0]
    if not isinstance(first, dict):
        raise RuntimeError("llama-server response had invalid embedding entry")

    embedding = first.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("llama-server response missing embedding vector")

    return len(embedding)


def _ensure_sqlite_vec_loaded(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def _read_existing_vec_dimension(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='page_embedding_vec'"
    ).fetchone()
    if row is None:
        return None

    sql = row[0]
    if not isinstance(sql, str):
        raise RuntimeError("Existing page_embedding_vec schema SQL is unavailable")

    match = _VEC_DIMENSION_PATTERN.search(sql)
    if match is None:
        raise RuntimeError("Existing page_embedding_vec schema has no parseable float dimension")

    return int(match.group(1))


def _ensure_vec_table(conn: sqlite3.Connection, dimensions: int) -> None:
    if dimensions <= 0:
        raise ValueError("embedding dimensions must be positive")

    existing_dimensions = _read_existing_vec_dimension(conn)
    if existing_dimensions is None:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS page_embedding_vec USING "
            f"vec0(embedding float[{dimensions}])"
        )
        return

    if existing_dimensions != dimensions:
        raise RuntimeError(
            "Existing page_embedding_vec dimension mismatch: "
            f"expected {dimensions}, found {existing_dimensions}"
        )


def _ensure_exactly_one_active_model(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    dimensions: int,
) -> None:
    if not model_name.strip():
        raise ValueError("embedding model_name must be non-empty")
    if dimensions <= 0:
        raise ValueError("embedding dimensions must be positive")

    total_models = int(conn.execute("SELECT COUNT(*) FROM embedding_models").fetchone()[0])
    active_models = int(
        conn.execute("SELECT COUNT(*) FROM embedding_models WHERE is_active = 1").fetchone()[0]
    )

    row = conn.execute(
        "SELECT model_id FROM embedding_models WHERE model_name = ?",
        (model_name,),
    ).fetchone()
    if row is None:
        is_active = 1 if total_models == 0 or active_models == 0 else 0
        conn.execute(
            "INSERT INTO embedding_models (model_name, dimensions, is_active) VALUES (?, ?, ?)",
            (model_name, dimensions, is_active),
        )
    else:
        conn.execute(
            "UPDATE embedding_models SET dimensions = ? WHERE model_name = ?",
            (dimensions, model_name),
        )

    conn.execute(
        "UPDATE embedding_models "
        "SET is_active = CASE WHEN model_name = ? THEN 1 ELSE 0 END",
        (model_name,),
    )


def init_schema(
    conn: sqlite3.Connection,
    *,
    embedding_dimensions: int | None = None,
    embedding_model_name: str | None = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    embedding_timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    embedding_api_key: str | None = None,
    embedding_probe_text: str = DEFAULT_EMBEDDING_PROBE_TEXT,
) -> None:
    """Initialize the v1 schema on the given connection. Idempotent."""
    conn.execute("PRAGMA foreign_keys = ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS source_profiles (
          source_profile_id INTEGER PRIMARY KEY,
          browser TEXT NOT NULL,
          profile_key TEXT NOT NULL UNIQUE,
          label TEXT
        );

        CREATE TABLE IF NOT EXISTS pages (
          page_id INTEGER PRIMARY KEY,
          url TEXT NOT NULL UNIQUE,
          host TEXT NOT NULL,
          title TEXT NOT NULL,
          first_seen_us INTEGER NOT NULL,
          last_seen_us INTEGER NOT NULL,
          normalization_version INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS visits (
          visit_id INTEGER PRIMARY KEY,
          page_id INTEGER NOT NULL REFERENCES pages(page_id) ON DELETE CASCADE,
          visited_at_us INTEGER NOT NULL,
          title_snapshot TEXT NOT NULL,
          source_profile_id INTEGER NOT NULL REFERENCES source_profiles(source_profile_id) ON DELETE CASCADE,
          source_visit_id INTEGER NOT NULL,
          UNIQUE (source_profile_id, source_visit_id)
        );

        CREATE TABLE IF NOT EXISTS bookmark_folders (
          folder_id INTEGER PRIMARY KEY,
          parent_folder_id INTEGER REFERENCES bookmark_folders(folder_id) ON DELETE CASCADE,
          title TEXT NOT NULL,
          position INTEGER NOT NULL,
          created_at_us INTEGER NOT NULL,
          updated_at_us INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bookmarks (
          bookmark_id INTEGER PRIMARY KEY,
          page_id INTEGER NOT NULL REFERENCES pages(page_id) ON DELETE CASCADE,
          folder_id INTEGER REFERENCES bookmark_folders(folder_id) ON DELETE SET NULL,
          title_override TEXT,
          notes TEXT,
          status TEXT NOT NULL,
          deleted_at_us INTEGER,
          created_at_us INTEGER NOT NULL,
          updated_at_us INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_pages_last_seen ON pages(last_seen_us DESC);
        CREATE INDEX IF NOT EXISTS idx_pages_host_last_seen ON pages(host, last_seen_us DESC);
        CREATE INDEX IF NOT EXISTS idx_visits_page_time ON visits(page_id, visited_at_us DESC);
        CREATE INDEX IF NOT EXISTS idx_visits_time ON visits(visited_at_us DESC);
        CREATE INDEX IF NOT EXISTS idx_bookmarks_status_updated ON bookmarks(status, updated_at_us DESC);
        CREATE INDEX IF NOT EXISTS idx_bookmarks_folder_updated ON bookmarks(folder_id, updated_at_us DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
          title,
          url,
          bookmark_text,
          content=''
        );

        CREATE TABLE IF NOT EXISTS embedding_models (
          model_id INTEGER PRIMARY KEY,
          model_name TEXT NOT NULL UNIQUE,
          dimensions INTEGER NOT NULL CHECK (dimensions > 0),
          is_active INTEGER NOT NULL CHECK (is_active IN (0, 1))
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_embedding_models_one_active
          ON embedding_models(is_active) WHERE is_active = 1;

        CREATE TRIGGER IF NOT EXISTS trg_embedding_models_exactly_one_active_insert
        AFTER INSERT ON embedding_models
        BEGIN
          SELECT CASE
            WHEN (
              SELECT COUNT(*) FROM embedding_models
            ) > 0 AND (
              SELECT COUNT(*) FROM embedding_models WHERE is_active = 1
            ) = 0
            THEN RAISE(ABORT, 'embedding_models must have exactly one active model')
          END;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_embedding_models_exactly_one_active_update
        AFTER UPDATE OF is_active ON embedding_models
        BEGIN
          SELECT CASE
            WHEN (
              SELECT COUNT(*) FROM embedding_models
            ) > 0 AND (
              SELECT COUNT(*) FROM embedding_models WHERE is_active = 1
            ) = 0
            THEN RAISE(ABORT, 'embedding_models must have exactly one active model')
          END;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_embedding_models_exactly_one_active_delete
        AFTER DELETE ON embedding_models
        BEGIN
          SELECT CASE
            WHEN (
              SELECT COUNT(*) FROM embedding_models
            ) > 0 AND (
              SELECT COUNT(*) FROM embedding_models WHERE is_active = 1
            ) = 0
            THEN RAISE(ABORT, 'embedding_models must have exactly one active model')
          END;
        END;

        CREATE TABLE IF NOT EXISTS page_embeddings (
          page_id INTEGER PRIMARY KEY REFERENCES pages(page_id) ON DELETE CASCADE,
          model_id INTEGER NOT NULL REFERENCES embedding_models(model_id) ON DELETE RESTRICT,
          content_hash TEXT NOT NULL,
          updated_at_us INTEGER NOT NULL
        );
    """)

    server_model_name: str | None = None
    server_dimensions: int | None = None
    should_fetch_server_info = (
        embedding_dimensions is None
        or embedding_model_name is None
        or not embedding_model_name.strip()
    )
    if should_fetch_server_info:
        server_model_name, server_dimensions = fetch_server_model_info(
            base_url=embedding_base_url,
            timeout_seconds=embedding_timeout_seconds,
            api_key=embedding_api_key,
        )

    if embedding_model_name is None or not embedding_model_name.strip():
        if server_model_name is None:
            raise RuntimeError("Unable to resolve embedding model name from llama-server")
        resolved_model_name = server_model_name
    else:
        resolved_model_name = _sanitize_model_identifier(embedding_model_name)

    if embedding_dimensions is not None:
        dimensions = int(embedding_dimensions)
    elif server_dimensions is not None:
        dimensions = int(server_dimensions)
    else:
        dimensions = discover_embedding_dimensions(
            base_url=embedding_base_url,
            model_name=embedding_model_name,
            timeout_seconds=embedding_timeout_seconds,
            api_key=embedding_api_key,
            probe_text=embedding_probe_text,
        )

    _ensure_sqlite_vec_loaded(conn)
    _ensure_vec_table(conn, dimensions)
    _ensure_exactly_one_active_model(
        conn,
        model_name=resolved_model_name,
        dimensions=dimensions,
    )
