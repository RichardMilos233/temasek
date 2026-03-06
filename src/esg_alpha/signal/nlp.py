from __future__ import annotations

import pandas as pd

from esg_alpha.models import ProcessedDataBundle
from esg_alpha.signal.base import SignalBuilder
from esg_alpha.utils import cross_sectional_zscore


class LexiconNLPESGSignalBuilder(SignalBuilder):
    """Lightweight sentiment signal from ESG-related headline text."""

    POSITIVE_WORDS = {
        "improvement",
        "improved",
        "favorable",
        "progress",
        "reduction",
        "upgrade",
        "strong",
    }
    NEGATIVE_WORDS = {
        "scrutiny",
        "setback",
        "penalty",
        "fine",
        "downgrade",
        "breach",
        "concern",
    }

    def __init__(self, smoothing_window: int = 5) -> None:
        self.smoothing_window = smoothing_window

    def build(self, data: ProcessedDataBundle) -> pd.DataFrame:
        empty_signal = pd.DataFrame(0.0, index=data.returns.index, columns=data.returns.columns)
        if data.news is None or data.news.empty:
            return empty_signal

        news = data.news.copy()
        news["date"] = pd.to_datetime(news["date"]).dt.normalize()
        news = news[news["ticker"].isin(data.returns.columns)]
        if news.empty:
            return empty_signal

        news["score"] = news["text"].astype(str).map(self._score_text)
        signal = news.groupby(["date", "ticker"])["score"].mean().unstack("ticker")
        signal = signal.reindex(index=data.returns.index, columns=data.returns.columns).fillna(0.0)

        if self.smoothing_window > 1:
            signal = signal.rolling(self.smoothing_window, min_periods=1).mean()

        return cross_sectional_zscore(signal)

    def _score_text(self, text: str) -> float:
        tokens = {token.strip(".,;:!?()[]{}\"").lower() for token in text.split()}
        pos_hits = len(tokens.intersection(self.POSITIVE_WORDS))
        neg_hits = len(tokens.intersection(self.NEGATIVE_WORDS))
        return float(pos_hits - neg_hits)


class FinBERTNewsSignalBuilder(SignalBuilder):
    """Transformer-based sentiment signal from ESG-related headlines."""

    LABEL_TO_SCORE = {
        "POSITIVE": 1.0,
        "NEUTRAL": 0.0,
        "NEGATIVE": -1.0,
        "Positive": 1.0,
        "Neutral": 0.0,
        "Negative": -1.0,
    }

    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        batch_size: int = 16,
        device: int = -1,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ImportError(
                    "FinBERT signal requires 'transformers' and a compatible torch install."
                ) from exc

            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device,
            )
        return self._classifier

    def build(self, data: ProcessedDataBundle) -> pd.DataFrame:
        empty_signal = pd.DataFrame(0.0, index=data.returns.index, columns=data.returns.columns)
        if data.news is None or data.news.empty:
            return empty_signal

        news = data.news.copy()
        news["date"] = pd.to_datetime(news["date"]).dt.normalize()
        news = news[news["ticker"].isin(data.returns.columns)]
        if news.empty:
            return empty_signal

        classifier = self._get_classifier()
        texts = news["text"].astype(str).tolist()
        predictions = classifier(texts, batch_size=self.batch_size, truncation=True)

        news["sentiment"] = [
            self.LABEL_TO_SCORE.get(prediction.get("label", "NEUTRAL"), 0.0)
            for prediction in predictions
        ]

        signal = news.groupby(["date", "ticker"])["sentiment"].mean().unstack("ticker")
        signal = signal.reindex(index=data.returns.index, columns=data.returns.columns).fillna(0.0)

        return cross_sectional_zscore(signal)
