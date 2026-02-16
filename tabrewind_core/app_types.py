from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_app_types_module() -> object:
    module_path = Path(__file__).resolve().parents[1] / "types.py"
    spec = importlib.util.spec_from_file_location("tabrewind_app_types", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load application types module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_APP_TYPES = _load_app_types_module()

AppConfig = _APP_TYPES.AppConfig
EncodedEntryPreview = _APP_TYPES.EncodedEntryPreview
HistoryEntry = _APP_TYPES.HistoryEntry
IngestSummary = _APP_TYPES.IngestSummary
ProfileStore = _APP_TYPES.ProfileStore


__all__ = [
    "AppConfig",
    "EncodedEntryPreview",
    "HistoryEntry",
    "IngestSummary",
    "ProfileStore",
]
