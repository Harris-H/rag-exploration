"""Shared knowledge base loading and caching."""

import json
import os
from pathlib import Path

_KB_CACHE: list[dict] | None = None

KB_PATH = Path(__file__).resolve().parent.parent.parent / "sample_docs" / "knowledge_base.json"


def load_knowledge_base() -> list[dict]:
    """Load knowledge base from JSON, caching globally after first call."""
    global _KB_CACHE
    if _KB_CACHE is not None:
        return _KB_CACHE
    with open(KB_PATH, "r", encoding="utf-8") as f:
        _KB_CACHE = json.load(f)
    return _KB_CACHE


def get_doc_count() -> int:
    return len(load_knowledge_base())
