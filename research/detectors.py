"""Детекторы ИИ-текста: перплексия n-gram LM и локальный ONNX-детектор (toiletsandpaper).

Перплексия — собственная метрика (низкая перплексия = «ИИ-подобный» текст).
OnnxDetector — бесплатная локальная модель для русского (MIT), API не нужен.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rankers import _corpus_sentences, _words, safe_logscore, train_ngram  # noqa: E402


class PerplexityDetector:
    """Перплексия текста под n-граммной моделью, обученной на человеческом корпусе."""

    name = "perplexity"

    def __init__(self, corpus: str | Path, order: int = 3) -> None:
        self.order = order
        self._model = train_ngram(_corpus_sentences(corpus), order)

    def score(self, text: str) -> float:
        """Перплексия текста: меньше — предсказуемее (более «ИИ»)."""
        words = _words(text)
        if len(words) < self.order:
            return float("inf")
        pad = self.order - 1
        context = ["<s>"] * pad
        total = 0.0
        for word in words:
            total += safe_logscore(self._model, word, tuple(context[-pad:])) if pad else 0.0
            context.append(word)
        return 2.0 ** min(-total / len(words), 30.0)


class OnnxDetector:
    """Вероятность, что русский текст написала LLM: toiletsandpaper/russian-ai-text-detector-*.

    Тонкая обёртка над app.critic.AICritic (общий код детектора живёт в app/).
    """

    name = "onnx"

    def __init__(self, model_id: str = "toiletsandpaper/russian-ai-text-detector-bert") -> None:
        from app.critic import AICritic

        self._critic = AICritic(model_id=model_id)

    def score(self, text: str) -> float:
        """Вероятность ИИ-происхождения текста (0..1), порог 0.5."""
        return self._critic.score(text)
