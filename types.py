from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HistoryEntry:
    browser: str
    profile: str
    title: str
    url: str
    canonical_url: str
    last_visit_date: int


@dataclass(frozen=True)
class ProfileStore:
    browser: str
    db_path: Path


@dataclass(frozen=True)
class EncodedEntryPreview:
    entry: HistoryEntry
    vector_preview: list[float]
    llm_tags: list[str] | None


@dataclass
class LlamaConfig:
    base_url: str = "http://127.0.0.1:8080"
    model: str = "embeddinggemma"
    api_key: str = "no-key"
    request_timeout_seconds: float = 30.0


@dataclass
class ProfilesConfig:
    zen_root: str = "~/Library/Application Support/Zen/Profiles"
    firefox_root: str = "~/Library/Application Support/Firefox/Profiles"


@dataclass
class IngestConfig:
    browsers: list[str] = field(default_factory=lambda: ["zen", "firefox"])
    domain_rules: list[str] = field(default_factory=list)
    profile_workers: int = 4
    embedding_workers: int = 4
    embedding_batch_size: int = 8
    limit_per_profile: int | None = None
    max_items: int | None = None
    preview_dims: int = 10
    llm_tags: list[str] = field(default_factory=list)


@dataclass
class AppConfig:
    config_version: int = 1
    llama: LlamaConfig = field(default_factory=LlamaConfig)
    profiles: ProfilesConfig = field(default_factory=ProfilesConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)


@dataclass(frozen=True)
class IngestSummary:
    discovered_profile_count: int
    loaded_row_count: int
    deduped_row_count: int
    effective_embedding_workers: int
    previews: list[EncodedEntryPreview]
    load_errors: list[str]
    embed_errors: list[str]
