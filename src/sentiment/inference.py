"""
FinBERT sentiment inference.

Returns continuous scores in [-1, 1]:
  +1 = maximally positive, -1 = maximally negative.

Computed as: score = p_positive - p_negative
where p_* are softmax probabilities from ProsusAI/finbert.

The positive/negative logit positions are resolved from the model's own
``config.id2label`` rather than hardcoded, because FinBERT's label order is
``{0: positive, 1: negative, 2: neutral}`` (not the intuitive neg/neutral/pos).
Reading the mapping makes the score correct regardless of label ordering and
prevents silent drift if the checkpoint changes.
"""

from __future__ import annotations

import torch
from transformers import BertForSequenceClassification, BertTokenizer

_MODEL_NAME = "ProsusAI/finbert"
_tokenizer: BertTokenizer | None = None
_model: BertForSequenceClassification | None = None
_pos_idx: int | None = None
_neg_idx: int | None = None


def _label_indices(model: BertForSequenceClassification) -> tuple[int, int]:
    """Return (positive_idx, negative_idx) resolved from ``config.id2label``."""
    global _pos_idx, _neg_idx
    if _pos_idx is None or _neg_idx is None:
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        try:
            _pos_idx, _neg_idx = label2id["positive"], label2id["negative"]
        except KeyError as exc:  # pragma: no cover - guards against odd checkpoints
            raise RuntimeError(
                f"FinBERT config.id2label missing positive/negative labels: {id2label}"
            ) from exc
    return _pos_idx, _neg_idx


def _load_model() -> tuple[BertTokenizer, BertForSequenceClassification]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = BertTokenizer.from_pretrained(_MODEL_NAME)
        _model = BertForSequenceClassification.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def score_text(text: str, max_length: int = 512) -> float:
    """
    Return sentiment score in [-1, 1] for a single text string.
    Positive = bullish, negative = bearish.
    """
    tokenizer, model = _load_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze()
    pos, neg = _label_indices(model)
    return float(probs[pos] - probs[neg])


def score_batch(texts: list[str], max_length: int = 512) -> list[float]:
    """Score a batch of texts. More efficient than calling score_text in a loop."""
    tokenizer, model = _load_model()
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)  # (N, 3)
    pos, neg = _label_indices(model)
    scores = probs[:, pos] - probs[:, neg]  # p_pos - p_neg
    return scores.tolist()
