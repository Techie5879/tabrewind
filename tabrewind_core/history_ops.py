from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

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

from .app_types import AppConfig, EncodedEntryPreview, HistoryEntry, IngestSummary, ProfileStore
from .config_store import SUPPORTED_BROWSERS
from .domain_policy import compile_domain_rules, resolve_domain_policy
from .vectorization import VectorizationClient, fetch_server_slots


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


def chunk_entries(entries: list[object], batch_size: int) -> list[tuple[int, list[object]]]:
    chunks: list[tuple[int, list[object]]] = []
    for start in range(0, len(entries), batch_size):
        chunks.append((start, entries[start : start + batch_size]))
    return chunks


def embed_batch(
    client: VectorizationClient,
    entries: list[HistoryEntry],
    preview_dims: int,
    llm_tags: list[str] | None,
) -> list[EncodedEntryPreview]:
    inputs = [build_embedding_input(entry) for entry in entries]
    vectors = client.encode_many(inputs)
    previews: list[EncodedEntryPreview] = []
    for entry, vector in zip(entries, vectors, strict=True):
        previews.append(
            EncodedEntryPreview(
                entry=entry,
                vector_preview=vector[:preview_dims],
                llm_tags=llm_tags,
            )
        )
    return previews


def run_ingest(config: AppConfig) -> IngestSummary:
    console = Console(stderr=False)
    compiled_domain_rules = compile_domain_rules(config.ingest.domain_rules)
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
        if compiled_domain_rules:
            filtered_entries: list[HistoryEntry] = []
            for entry in deduped_entries:
                decision = resolve_domain_policy(entry.canonical_url, compiled_domain_rules)
                if decision.allowed:
                    filtered_entries.append(entry)
            deduped_entries = filtered_entries

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

        vector_client = VectorizationClient(
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
        previews: list[EncodedEntryPreview | None] = [None] * len(deduped_entries)

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


__all__ = [
    "build_embedding_input",
    "chunk_entries",
    "dedupe_entries",
    "discover_places_files",
    "embed_batch",
    "load_history_entries",
    "normalize_url",
    "run_ingest",
]
