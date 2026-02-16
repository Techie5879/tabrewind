from __future__ import annotations

import tomllib
from pathlib import Path

from .app_types import AppConfig
from .domain_policy import compile_domain_rules


CONFIG_FILE_NAME = "config.toml"
SUPPORTED_BROWSERS = {"zen", "firefox"}


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
        f"domain_rules = {_toml_string_array(config.ingest.domain_rules)}",
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
        domain_rules = _as_string_list(ingest_data.get("domain_rules"), [])
        compile_domain_rules(domain_rules)
        config.ingest.domain_rules = domain_rules

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
    if key == "ingest.domain_rules":
        candidate_rules = parse_csv_list(value)
        compile_domain_rules(candidate_rules)
        config.ingest.domain_rules = candidate_rules
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
        "ingest.domain_rules",
    ]
    raise ValueError(
        f"Unsupported config key '{key}'. Supported keys: {', '.join(supported_keys)}"
    )


__all__ = [
    "CONFIG_FILE_NAME",
    "SUPPORTED_BROWSERS",
    "init_config",
    "load_config",
    "parse_csv_list",
    "render_config_toml",
    "save_config",
    "set_config_key",
]
