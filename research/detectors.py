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

    Модель скачивается с HF один раз (int8 ONNX ~30 МБ) и работает на CPU.
    """

    name = "onnx"

    def __init__(self, model_id: str = "toiletsandpaper/russian-ai-text-detector-bert") -> None:
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        self._np = np
        path = snapshot_download(model_id, allow_patterns=["inference.json", "model.onnx*", "tokenizer.json"])
        with open(f"{path}/inference.json", encoding="utf-8") as fh:
            self.spec = json.load(fh)
        self._session = ort.InferenceSession(f"{path}/{self.spec['file']}", providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(f"{path}/{self.spec['tokenizer']}")
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def score(self, text: str) -> float:
        """Вероятность ИИ-происхождения текста (0..1), порог 0.5."""
        np = self._np
        spec = self.spec
        text = text[: spec["max_chars"]]
        for rule in spec["normalize"]:
            text = re.sub(rule["pattern"], rule["replacement"], text, flags=re.MULTILINE)
        ids = self._tokenizer.encode(text.strip(), add_special_tokens=False).ids
        prefix, suffix = [spec["cls_id"]], [spec["sep_id"]]
        width = spec.get("read_length", spec["max_length"]) - len(prefix) - len(suffix)
        windows = [ids[i : i + width] for i in range(0, max(len(ids), 1), width)][: spec["max_windows"]]
        windows = [[*prefix, *w, *suffix] for w in windows]
        batch = np.full((len(windows), max(map(len, windows))), spec["pad_id"], dtype=np.int64)
        mask = np.zeros_like(batch)
        for k, w in enumerate(windows):
            batch[k, : len(w)] = w
            mask[k, : len(w)] = 1
        logits = self._session.run([spec["output"]], {"input_ids": batch, "attention_mask": mask})[0]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        return float((p[:, spec["labels"].index("ai")] / p.sum(axis=1)).mean())
