"""
FinBERT sentiment inference.

Returns continuous scores in [-1, 1]:
  +1 = maximally positive, -1 = maximally negative.

Computed as: score = p_positive - p_negative
where p_* are softmax probabilities from ProsusAI/finbert.
"""

from __future__ import annotations

import torch
from transformers import BertTokenizer, BertForSequenceClassification

_MODEL_NAME = "ProsusAI/finbert"
_tokenizer: BertTokenizer | None = None
_model: BertForSequenceClassification | None = None


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
    probs = torch.softmax(logits, dim=1).squeeze()  # [neg, neutral, pos]
    return float(probs[2] - probs[0])


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
    scores = probs[:, 2] - probs[:, 0]   # p_pos - p_neg
    return scores.tolist()
