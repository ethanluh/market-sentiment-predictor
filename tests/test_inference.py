"""Tests for FinBERT label resolution (no torch / no model download).

``src/sentiment/inference.py`` imports ``torch``/``transformers`` at module
load, which CI deliberately omits (see requirements-dev.txt and the
conftest heavy-import guard). The tensor-scoring paths therefore can't run
here — but ``_label_indices`` is pure dict logic that only reads
``model.config.id2label``. We inject light fakes for the heavy modules (the
same sys.modules-injection pattern used in test_news_archive.py) so the
import succeeds, then exercise the label-order resolution that the docstring
flags as historically sign-bug-prone, plus its bad-checkpoint guard.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def inference():
    """Import src.sentiment.inference with torch/transformers faked out."""
    saved = {k: sys.modules.get(k) for k in ("torch", "transformers", "src.sentiment.inference")}

    sys.modules["torch"] = types.ModuleType("torch")
    fake_tf = types.ModuleType("transformers")
    fake_tf.BertForSequenceClassification = type("BertForSequenceClassification", (), {})
    fake_tf.BertTokenizer = type("BertTokenizer", (), {})
    sys.modules["transformers"] = fake_tf
    sys.modules.pop("src.sentiment.inference", None)

    import src.sentiment.inference as module

    yield module

    # Restore sys.modules so the heavy-dep keys don't leak into other tests.
    for key, value in saved.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value


def _model(id2label: dict[int, str]):
    """A stand-in exposing only ``config.id2label``, like a HF model."""
    config = type("Config", (), {"id2label": id2label})()
    return type("Model", (), {"config": config})()


def test_resolves_finbert_label_order(inference):
    # FinBERT's actual (non-intuitive) ordering.
    model = _model({0: "positive", 1: "negative", 2: "neutral"})
    assert inference._label_indices(model) == (0, 1)


def test_resolution_is_order_independent(inference):
    # A checkpoint with a different ordering must still map correctly.
    model = _model({0: "neutral", 1: "negative", 2: "positive"})
    assert inference._label_indices(model) == (2, 1)


def test_labels_are_case_insensitive(inference):
    model = _model({0: "POSITIVE", 1: "Negative", 2: "Neutral"})
    assert inference._label_indices(model) == (0, 1)


def test_missing_labels_raise_runtime_error(inference):
    model = _model({0: "bullish", 1: "bearish", 2: "flat"})
    with pytest.raises(RuntimeError, match="missing positive/negative"):
        inference._label_indices(model)


def test_indices_cached_after_first_call(inference):
    model = _model({0: "positive", 1: "negative", 2: "neutral"})
    assert inference._label_indices(model) == (0, 1)
    # A later call ignores a mutated config — the resolution is cached.
    model.config.id2label = {0: "negative", 1: "positive", 2: "neutral"}
    assert inference._label_indices(model) == (0, 1)
