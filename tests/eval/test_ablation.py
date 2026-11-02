import pytest

from src.eval.ablation import FEATURE_GROUPS, ablation_configs, neutralize
from src.eval.synthetic import make_synthetic
from src.prediction.features import FEATURE_NAMES


def test_neutralize_sets_constants_without_mutating_input() -> None:
    f = make_synthetic(n_days=120).features
    before = f.copy()
    out = neutralize(f, ["sentiment", "regime"])
    assert (out["sentiment_agg"] == 0.0).all() and (out["regime_label"] == -1.0).all()
    assert out["momentum_5d"].equals(f["momentum_5d"])
    assert f.equals(before)


def test_unknown_group_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        neutralize(make_synthetic(n_days=120).features, ["nope"])


def test_configs_cover_price_only_each_group_and_full() -> None:
    cfg = ablation_configs()
    assert cfg["price_only"] == list(FEATURE_GROUPS) and cfg["full"] == []
    for g in FEATURE_GROUPS:
        assert g not in cfg[f"price+{g}"] and len(cfg[f"price+{g}"]) == len(FEATURE_GROUPS) - 1


def test_groups_reference_real_feature_names() -> None:
    assert {c for cols in FEATURE_GROUPS.values() for c in cols} <= set(FEATURE_NAMES)
