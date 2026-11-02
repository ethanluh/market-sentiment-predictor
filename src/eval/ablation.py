"""Feature-group ablation: neutralise groups of features and re-evaluate."""

from __future__ import annotations

import pandas as pd

# Groups of ``FEATURE_NAMES`` columns that come from one data source.
FEATURE_GROUPS: dict[str, list[str]] = {
    "sentiment": ["sentiment_agg", "estimated_lag_hours"],
    "sector": ["sector_proximity"],
    "trends": ["search_interest_zscore"],
    "insider": ["insider_flow_npr"],
    "regime": ["regime_label"],
}

# Value that marks each neutralised column as "no information" (the same neutral
# values the live pipeline emits when a source is disabled).
_NEUTRAL: dict[str, float] = {
    "sentiment_agg": 0.0,
    "estimated_lag_hours": 6.0,
    "sector_proximity": 0.0,
    "search_interest_zscore": 0.0,
    "insider_flow_npr": 0.0,
    "regime_label": -1.0,
}


def neutralize(features: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    """Copy of ``features`` with every column in ``groups`` set to its neutral constant."""
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown feature groups {unknown}; choose from {list(FEATURE_GROUPS)}")
    out = features.copy()
    for g in groups:
        for col in FEATURE_GROUPS[g]:
            out[col] = _NEUTRAL[col]
    return out


def ablation_configs() -> dict[str, list[str]]:
    """
    Name -> groups to neutralise: ``price_only``, ``price+<group>`` for each group,
    and ``full``. The first answers "does any non-price source help at all?" and
    each ``price+<group>`` isolates one source.
    """
    all_groups = list(FEATURE_GROUPS)
    configs = {"price_only": all_groups}
    for g in all_groups:
        configs[f"price+{g}"] = [x for x in all_groups if x != g]
    configs["full"] = []
    return configs
