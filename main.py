from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def _load_app_types_module() -> object:
    module_path = Path(__file__).with_name("types.py")
    spec = importlib.util.spec_from_file_location("tabrewind_app_types", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load application types module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_APP_TYPES = _load_app_types_module()
AppConfig = _APP_TYPES.AppConfig
HistoryEntry = _APP_TYPES.HistoryEntry
IngestSummary = _APP_TYPES.IngestSummary
ProfileStore = _APP_TYPES.ProfileStore
VectorPreview = _APP_TYPES.VectorPreview
WorkerTiming = _APP_TYPES.WorkerTiming
WorkerSweepResult = _APP_TYPES.WorkerSweepResult

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'rich'. Install project dependencies with uv sync."
    ) from exc


CONFIG_FILE_NAME = "config.toml"
SUPPORTED_BROWSERS = {"zen", "firefox"}


class LlamaCppVectorClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    def _post_json(self, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            url=f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama-server request failed with HTTP {exc.code}: {details}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to connect to llama-server: {exc.reason}") from exc

        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise RuntimeError("llama-server returned unexpected JSON response shape")
        return parsed

    def encode_text(self, text: str) -> list[float]:
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "input": texts,
            "model": self.model,
            "encoding_format": "float",
        }
        data = self._post_json("/v1/embeddings", payload)
        items = data.get("data")
        if not isinstance(items, list):
            raise RuntimeError("llama-server response did not include embeddings data")

        vectors: list[list[float] | None] = [None] * len(texts)
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                raise RuntimeError("llama-server response had unexpected embeddings entry")
            index_raw = item.get("index", position)
            try:
                index = int(index_raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("llama-server response returned invalid embedding index") from exc

            if index < 0 or index >= len(vectors):
                raise RuntimeError("llama-server response index out of expected range")

            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError("llama-server response missing embedding vector")
            vectors[index] = [float(value) for value in embedding]

        if any(vector is None for vector in vectors):
            raise RuntimeError("llama-server response omitted one or more embeddings")

        return [vector for vector in vectors if vector is not None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tabrewind",
        description="Config-first CLI for local browser history ingestion.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(CONFIG_FILE_NAME),
        help="Path to config TOML file (default: ./config.toml).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create default config.toml.")
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config.toml.",
    )

    config_cmd = subparsers.add_parser("config", help="Inspect or update config values.")
    config_subparsers = config_cmd.add_subparsers(dest="config_command", required=True)
    config_subparsers.add_parser("show", help="Print resolved config.")
    config_subparsers.add_parser("path", help="Print config path.")

    config_set = config_subparsers.add_parser("set", help="Set a config key and save.")
    config_set.add_argument("key", help="Dot-path key, e.g. llama.base_url")
    config_set.add_argument("value", help="New value")

    ingest = subparsers.add_parser(
        "ingest",
        help="Load browser history and print embedding vector previews.",
    )
    ingest.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional one-shot override for max deduped items.",
    )

    bench = subparsers.add_parser(
        "bench",
        help="Run a local concurrency sweep benchmark.",
    )
    bench.add_argument(
        "--sentences",
        type=int,
        default=20,
        help="Number of sentences to benchmark.",
    )
    bench.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Maximum worker count in sweep (sweep always includes workers=0 baseline).",
    )

    return parser.parse_args()


def _escape_toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_string_array(values: list[str]) -> str:
    return "[" + ", ".join(_escape_toml_string(item) for item in values) + "]"


def render_config_toml(config: AppConfig) -> str:
    limit_per_profile = 0 if config.ingest.limit_per_profile is None else config.ingest.limit_per_profile
    max_items = 0 if config.ingest.max_items is None else config.ingest.max_items
    lines = [
        "config_version = 1",
        "",
        "[llama]",
        f"base_url = {_escape_toml_string(config.llama.base_url)}",
        f"model = {_escape_toml_string(config.llama.model)}",
        f"api_key = {_escape_toml_string(config.llama.api_key)}",
        f"request_timeout_seconds = {float(config.llama.request_timeout_seconds)}",
        "",
        "[profiles]",
        f"zen_root = {_escape_toml_string(config.profiles.zen_root)}",
        f"firefox_root = {_escape_toml_string(config.profiles.firefox_root)}",
        "",
        "[ingest]",
        f"browsers = {_toml_string_array(config.ingest.browsers)}",
        f"profile_workers = {int(config.ingest.profile_workers)}",
        f"embedding_workers = {int(config.ingest.embedding_workers)}",
        f"embedding_batch_size = {int(config.ingest.embedding_batch_size)}",
        f"limit_per_profile = {int(limit_per_profile)}",
        f"max_items = {int(max_items)}",
        f"preview_dims = {int(config.ingest.preview_dims)}",
        f"llm_tags = {_toml_string_array(config.ingest.llm_tags)}",
        "",
    ]
    return "\n".join(lines)


def save_config(path: Path, config: AppConfig) -> None:
    path.write_text(render_config_toml(config), encoding="utf-8")


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_string_list(value: object, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return default
    output = [str(item).strip() for item in value if str(item).strip()]
    return output or default


def _coerce_limit(value: object) -> int | None:
    parsed = _as_int(value, 0)
    if parsed <= 0:
        return None
    return parsed


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Run `uv run main.py init` first."
        )

    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    config = AppConfig()

    config.config_version = _as_int(raw.get("config_version"), 1)

    llama_data = raw.get("llama", {})
    if isinstance(llama_data, dict):
        config.llama.base_url = str(llama_data.get("base_url", config.llama.base_url))
        config.llama.model = str(llama_data.get("model", config.llama.model))
        config.llama.api_key = str(llama_data.get("api_key", config.llama.api_key))
        config.llama.request_timeout_seconds = _as_float(
            llama_data.get("request_timeout_seconds"),
            config.llama.request_timeout_seconds,
        )

    profile_data = raw.get("profiles", {})
    if isinstance(profile_data, dict):
        config.profiles.zen_root = str(profile_data.get("zen_root", config.profiles.zen_root))
        config.profiles.firefox_root = str(
            profile_data.get("firefox_root", config.profiles.firefox_root)
        )

    ingest_data = raw.get("ingest", {})
    if isinstance(ingest_data, dict):
        browsers = _as_string_list(ingest_data.get("browsers"), config.ingest.browsers)
        cleaned_browsers = [browser for browser in browsers if browser in SUPPORTED_BROWSERS]
        config.ingest.browsers = cleaned_browsers or config.ingest.browsers
        config.ingest.profile_workers = max(
            1,
            _as_int(ingest_data.get("profile_workers"), config.ingest.profile_workers),
        )
        config.ingest.embedding_workers = max(
            1,
            _as_int(ingest_data.get("embedding_workers"), config.ingest.embedding_workers),
        )
        config.ingest.embedding_batch_size = max(
            1,
            _as_int(
                ingest_data.get("embedding_batch_size"),
                config.ingest.embedding_batch_size,
            ),
        )
        config.ingest.limit_per_profile = _coerce_limit(ingest_data.get("limit_per_profile"))
        config.ingest.max_items = _coerce_limit(ingest_data.get("max_items"))
        config.ingest.preview_dims = max(
            1,
            _as_int(ingest_data.get("preview_dims"), config.ingest.preview_dims),
        )
        config.ingest.llm_tags = _as_string_list(ingest_data.get("llm_tags"), [])

    return config


def init_config(path: Path, force: bool) -> tuple[AppConfig, bool]:
    config = AppConfig()
    existed = path.exists()
    if existed and not force:
        return config, False

    save_config(path, config)
    return config, True


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def set_config_key(config: AppConfig, key: str, value: str) -> None:
    if key == "llama.base_url":
        config.llama.base_url = value.strip()
        return
    if key == "llama.model":
        config.llama.model = value.strip()
        return
    if key == "llama.api_key":
        config.llama.api_key = value.strip()
        return
    if key == "llama.request_timeout_seconds":
        config.llama.request_timeout_seconds = max(1.0, float(value))
        return
    if key == "profiles.zen_root":
        config.profiles.zen_root = value.strip()
        return
    if key == "profiles.firefox_root":
        config.profiles.firefox_root = value.strip()
        return
    if key == "ingest.browsers":
        browsers = parse_csv_list(value)
        invalid = [browser for browser in browsers if browser not in SUPPORTED_BROWSERS]
        if invalid:
            raise ValueError(f"Unsupported browser(s): {', '.join(invalid)}")
        config.ingest.browsers = browsers or config.ingest.browsers
        return
    if key == "ingest.profile_workers":
        config.ingest.profile_workers = max(1, int(value))
        return
    if key == "ingest.embedding_workers":
        config.ingest.embedding_workers = max(1, int(value))
        return
    if key == "ingest.embedding_batch_size":
        config.ingest.embedding_batch_size = max(1, int(value))
        return
    if key == "ingest.limit_per_profile":
        parsed = int(value)
        config.ingest.limit_per_profile = None if parsed <= 0 else parsed
        return
    if key == "ingest.max_items":
        parsed = int(value)
        config.ingest.max_items = None if parsed <= 0 else parsed
        return
    if key == "ingest.preview_dims":
        config.ingest.preview_dims = max(1, int(value))
        return
    if key == "ingest.llm_tags":
        config.ingest.llm_tags = parse_csv_list(value)
        return

    supported_keys = [
        "llama.base_url",
        "llama.model",
        "llama.api_key",
        "llama.request_timeout_seconds",
        "profiles.zen_root",
        "profiles.firefox_root",
        "ingest.browsers",
        "ingest.profile_workers",
        "ingest.embedding_workers",
        "ingest.embedding_batch_size",
        "ingest.limit_per_profile",
        "ingest.max_items",
        "ingest.preview_dims",
        "ingest.llm_tags",
    ]
    raise ValueError(
        f"Unsupported config key '{key}'. Supported keys: {', '.join(supported_keys)}"
    )


def discover_places_files(browser: str, profile_root: str) -> list[ProfileStore]:
    root_path = Path(profile_root).expanduser()
    if not root_path.exists() or not root_path.is_dir():
        return []

    stores = [
        ProfileStore(browser=browser, db_path=path)
        for path in sorted(root_path.glob("*/places.sqlite"))
        if path.is_file()
    ]
    return stores


def normalize_url(url: str) -> str | None:
    value = url.strip()
    if not value:
        return None

    try:
        parsed = urlsplit(value)
    except ValueError:
        return None

    scheme = parsed.scheme.lower()
    if not scheme:
        return None

    host = (parsed.hostname or "").lower()
    if scheme in {"http", "https"} and not host:
        return None

    netloc = parsed.netloc.lower()
    if host:
        try:
            port = parsed.port
        except ValueError:
            return None
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None
        userinfo = ""
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo = f"{userinfo}:{parsed.password}"
            userinfo = f"{userinfo}@"
        netloc = f"{userinfo}{host}"
        if port is not None:
            netloc = f"{netloc}:{port}"

    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def load_history_entries(
    store: ProfileStore,
    limit_per_profile: int | None,
) -> tuple[list[HistoryEntry], str | None]:
    uri = f"file:{store.db_path.as_posix()}?mode=ro&immutable=1"
    query = (
        "SELECT p.url, COALESCE(p.title, ''), COALESCE(p.last_visit_date, 0) "
        "FROM moz_places AS p "
        "WHERE p.url IS NOT NULL AND TRIM(p.url) != '' "
        "ORDER BY COALESCE(p.last_visit_date, 0) DESC, p.id DESC"
    )

    rows: list[HistoryEntry] = []
    profile_name = store.db_path.parent.name
    try:
        with sqlite3.connect(uri, uri=True, timeout=1.0) as conn:
            cursor = conn.cursor()
            if limit_per_profile is None:
                cursor.execute(query)
            else:
                cursor.execute(f"{query} LIMIT ?", (limit_per_profile,))

            for url_value, title_value, last_visit_date in cursor:
                canonical = normalize_url(url_value)
                if canonical is None:
                    continue

                title = title_value.strip() if isinstance(title_value, str) else ""
                if not title:
                    title = "(untitled)"

                rows.append(
                    HistoryEntry(
                        browser=store.browser,
                        profile=profile_name,
                        title=title,
                        url=url_value,
                        canonical_url=canonical,
                        last_visit_date=int(last_visit_date),
                    )
                )
    except sqlite3.Error as exc:
        return [], f"{store.browser}:{profile_name} ({store.db_path.name}) -> {exc}"

    return rows, None


def dedupe_entries(entries: list[HistoryEntry]) -> list[HistoryEntry]:
    by_url: dict[str, HistoryEntry] = {}
    for item in entries:
        existing = by_url.get(item.canonical_url)
        if existing is None:
            by_url[item.canonical_url] = item
            continue

        replacement_needed = False
        if item.last_visit_date > existing.last_visit_date:
            replacement_needed = True
        elif item.last_visit_date == existing.last_visit_date:
            if (item.browser, item.profile, item.url) < (
                existing.browser,
                existing.profile,
                existing.url,
            ):
                replacement_needed = True

        if replacement_needed:
            by_url[item.canonical_url] = item

    return sorted(
        by_url.values(),
        key=lambda item: (-item.last_visit_date, item.canonical_url, item.title),
    )


def build_embedding_input(entry: HistoryEntry) -> str:
    return f"title: {entry.title}\nurl: {entry.canonical_url}"


def fetch_server_slots(base_url: str, timeout_seconds: float) -> int | None:
    request = Request(url=f"{base_url.rstrip('/')}/slots", method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    if isinstance(data, list):
        return max(1, len(data))
    if isinstance(data, dict):
        total_slots = data.get("total_slots")
        try:
            return max(1, int(total_slots))
        except (TypeError, ValueError):
            return None
    return None


def chunk_entries(entries: list[HistoryEntry], batch_size: int) -> list[tuple[int, list[HistoryEntry]]]:
    chunks: list[tuple[int, list[HistoryEntry]]] = []
    for start in range(0, len(entries), batch_size):
        chunks.append((start, entries[start : start + batch_size]))
    return chunks


def embed_batch(
    client: LlamaCppVectorClient,
    entries: list[HistoryEntry],
    preview_dims: int,
    llm_tags: list[str] | None,
) -> list[VectorPreview]:
    inputs = [build_embedding_input(entry) for entry in entries]
    vectors = client.encode_many(inputs)
    previews: list[VectorPreview] = []
    for entry, vector in zip(entries, vectors, strict=True):
        previews.append(
            VectorPreview(
                entry=entry,
                vector_preview=vector[:preview_dims],
                llm_tags=llm_tags,
            )
        )
    return previews


def run_ingest(config: AppConfig) -> IngestSummary:
    console = Console(stderr=False)
    selected_browsers = [browser for browser in config.ingest.browsers if browser in SUPPORTED_BROWSERS]
    if not selected_browsers:
        selected_browsers = ["zen", "firefox"]

    discovered_profiles: list[ProfileStore] = []
    load_errors: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        discovery_task = progress.add_task(
            "Discovering browser profiles", total=len(selected_browsers)
        )
        for browser in selected_browsers:
            root = config.profiles.zen_root if browser == "zen" else config.profiles.firefox_root
            discovered_profiles.extend(discover_places_files(browser, root))
            progress.advance(discovery_task)

        if not discovered_profiles:
            return IngestSummary(
                discovered_profile_count=0,
                loaded_row_count=0,
                deduped_row_count=0,
                effective_embedding_workers=max(1, config.ingest.embedding_workers),
                previews=[],
                load_errors=[],
                embed_errors=[],
            )

        history_task = progress.add_task(
            "Loading history rows", total=len(discovered_profiles)
        )
        all_entries: list[HistoryEntry] = []
        with ThreadPoolExecutor(max_workers=max(1, config.ingest.profile_workers)) as executor:
            futures = {
                executor.submit(
                    load_history_entries,
                    store,
                    config.ingest.limit_per_profile,
                ): store
                for store in discovered_profiles
            }
            for future in as_completed(futures):
                rows, error = future.result()
                all_entries.extend(rows)
                if error:
                    load_errors.append(error)
                progress.advance(history_task)

        deduped_entries = dedupe_entries(all_entries)
        if config.ingest.max_items is not None:
            deduped_entries = deduped_entries[: max(0, config.ingest.max_items)]

        if not deduped_entries:
            return IngestSummary(
                discovered_profile_count=len(discovered_profiles),
                loaded_row_count=len(all_entries),
                deduped_row_count=0,
                effective_embedding_workers=max(1, config.ingest.embedding_workers),
                previews=[],
                load_errors=load_errors,
                embed_errors=[],
            )

        slots = fetch_server_slots(
            base_url=config.llama.base_url,
            timeout_seconds=config.llama.request_timeout_seconds,
        )
        configured_workers = max(1, config.ingest.embedding_workers)
        if slots is None:
            effective_workers = configured_workers
        else:
            effective_workers = max(1, min(configured_workers, slots))

        vector_client = LlamaCppVectorClient(
            base_url=config.llama.base_url,
            model=config.llama.model,
            timeout_seconds=config.llama.request_timeout_seconds,
            api_key=config.llama.api_key,
        )

        chunks = chunk_entries(
            deduped_entries,
            max(1, config.ingest.embedding_batch_size),
        )
        embed_task = progress.add_task(
            "Requesting embeddings", total=len(deduped_entries)
        )
        embed_errors: list[str] = []
        previews: list[VectorPreview | None] = [None] * len(deduped_entries)

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    embed_batch,
                    vector_client,
                    batch,
                    config.ingest.preview_dims,
                    config.ingest.llm_tags or None,
                ): start
                for start, batch in chunks
            }
            for future in as_completed(futures):
                start_index = futures[future]
                batch_size = min(
                    config.ingest.embedding_batch_size,
                    len(deduped_entries) - start_index,
                )
                try:
                    batch_previews = future.result()
                    for offset, preview in enumerate(batch_previews):
                        previews[start_index + offset] = preview
                except Exception as exc:
                    entry = deduped_entries[start_index]
                    embed_errors.append(
                        f"embedding failed for {entry.browser}:{entry.profile} {entry.canonical_url}: {exc}"
                    )
                progress.advance(embed_task, advance=batch_size)

    finalized_previews = [preview for preview in previews if preview is not None]
    return IngestSummary(
        discovered_profile_count=len(discovered_profiles),
        loaded_row_count=len(all_entries),
        deduped_row_count=len(deduped_entries),
        effective_embedding_workers=effective_workers,
        previews=finalized_previews,
        load_errors=load_errors,
        embed_errors=embed_errors,
    )


def _run_vectorization_with_workers(
    client: LlamaCppVectorClient,
    sentences: list[str],
    workers: int,
) -> tuple[list[list[float]], float]:
    start = time.perf_counter()
    if workers <= 0:
        vectors = [client.encode_text(sentence) for sentence in sentences]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(client.encode_text, sentence) for sentence in sentences]
            vectors = [future.result() for future in futures]
    elapsed_seconds = time.perf_counter() - start
    return vectors, elapsed_seconds


def benchmark_worker_sweep(
    client: LlamaCppVectorClient,
    sentences: list[str],
    max_workers: int,
) -> WorkerSweepResult:
    if not sentences:
        raise ValueError("Benchmark requires at least one sentence")

    baseline_vectors, baseline_seconds = _run_vectorization_with_workers(
        client,
        sentences,
        workers=0,
    )

    timings: list[WorkerTiming] = [
        WorkerTiming(
            workers=0,
            elapsed_seconds=baseline_seconds,
        )
    ]

    for workers in range(1, max(1, max_workers) + 1):
        vectors, elapsed_seconds = _run_vectorization_with_workers(client, sentences, workers)
        if len(vectors) != len(baseline_vectors):
            raise RuntimeError("Worker sweep returned mismatched vector counts")
        timings.append(WorkerTiming(workers=workers, elapsed_seconds=elapsed_seconds))

    vector_dimensions = len(baseline_vectors[0])
    return WorkerSweepResult(
        sentence_count=len(sentences),
        baseline_seconds=baseline_seconds,
        vector_dimensions=vector_dimensions,
        timings=timings,
    )


def run_init_command(args: argparse.Namespace, console: Console) -> int:
    config_path: Path = args.config
    _, did_write = init_config(config_path, force=args.force)
    if did_write:
        console.print(f"Wrote config to {config_path}")
    else:
        console.print(f"Config already exists at {config_path}. Use --force to overwrite.")
    return 0


def run_config_command(args: argparse.Namespace, console: Console) -> int:
    config_path: Path = args.config
    if args.config_command == "path":
        console.print(str(config_path))
        return 0

    config = load_config(config_path)
    if args.config_command == "show":
        console.print(render_config_toml(config), markup=False)
        return 0

    if args.config_command == "set":
        set_config_key(config, args.key, args.value)
        save_config(config_path, config)
        console.print(f"Updated {args.key} in {config_path}")
        return 0

    raise RuntimeError(f"Unknown config command: {args.config_command}")


def run_ingest_command(args: argparse.Namespace, console: Console) -> int:
    config = load_config(args.config)
    if args.max_items is not None:
        config.ingest.max_items = None if args.max_items <= 0 else args.max_items

    summary = run_ingest(config)
    if summary.discovered_profile_count == 0:
        console.print("No places.sqlite files were discovered for configured browsers.")
        return 0

    console.print(
        "Discovered "
        f"{summary.discovered_profile_count} profile DB(s); loaded {summary.loaded_row_count} rows; "
        f"deduped to {summary.deduped_row_count} rows; embedding workers {summary.effective_embedding_workers}."
    )

    if summary.load_errors:
        console.print("Profile read errors:")
        for message in summary.load_errors:
            console.print(f"- {message}")

    if summary.embed_errors:
        console.print("Embedding errors:")
        for message in summary.embed_errors:
            console.print(f"- {message}")

    for preview in summary.previews:
        vector_values = ", ".join(f"{value:.6f}" for value in preview.vector_preview)
        console.print(
            f"[{preview.entry.browser}:{preview.entry.profile}] {preview.entry.title} -> {preview.entry.canonical_url}"
        )
        if preview.llm_tags:
            console.print(f"tags: {', '.join(preview.llm_tags)}")
        console.print(f"vec[:{len(preview.vector_preview)}]: [{vector_values}]")

    return 0


def run_bench_command(args: argparse.Namespace, console: Console) -> int:
    config = load_config(args.config)
    max_workers = max(0, args.max_workers)

    sentences = [
        f"tabrewind benchmark sentence {index} with random-ish token {(index * 17) % 23}"
        for index in range(max(1, args.sentences))
    ]

    client = LlamaCppVectorClient(
        base_url=config.llama.base_url,
        model=config.llama.model,
        timeout_seconds=config.llama.request_timeout_seconds,
        api_key=config.llama.api_key,
    )
    result = benchmark_worker_sweep(client, sentences, max_workers=max_workers)

    console.print(f"Worker sweep: 0..{max_workers}")
    console.print(f"Sentences: {result.sentence_count}")
    console.print(f"Vector dimensions: {result.vector_dimensions}")
    console.print(f"Baseline (workers=0): {result.baseline_seconds:.3f}s")
    for timing in result.timings:
        speedup = (
            result.baseline_seconds / timing.elapsed_seconds
            if timing.elapsed_seconds > 0
            else 0.0
        )
        console.print(
            f"workers={timing.workers:>2} elapsed={timing.elapsed_seconds:.3f}s speedup={speedup:.2f}x"
        )
    return 0


def main() -> int:
    args = parse_args()
    console = Console(stderr=False)

    if args.command == "init":
        return run_init_command(args, console)
    if args.command == "config":
        return run_config_command(args, console)
    if args.command == "ingest":
        return run_ingest_command(args, console)
    if args.command == "bench":
        return run_bench_command(args, console)

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
