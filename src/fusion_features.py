"""Fusion feature names (model JSON) and feature_cache.csv column mapping."""

from __future__ import annotations

# Valid names for --features and model JSON feature_names
ALL_FUSION_FEATURES = ["Sv", "Sl", "Sb", "Sh", "Sa", "Sf"]


def cache_column(name: str) -> str:
    """CSV column name for a fusion feature."""
    if name == "Sf":
        return "Sf_pipeline"
    return name


def cache_load_keys() -> list[str]:
    """Columns to load from feature_cache.csv into a row dict."""
    return ["Sv", "Sl", "Sb", "Sh", "Sa", "Sf_pipeline"]
