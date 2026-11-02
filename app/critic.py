"""Детектор-критик: локальный RU AI-детектор toiletsandpaper (ONNX) для цикла правок.

Используется как критик в PolishTransformer: текст принимается, когда детектор
перестаёт помечать его как ИИ (скор < 0.5). Зависимости опциональные:
onnxruntime, tokenizers, huggingface_hub.
"""
from __future__ import annotations

import json
import os
import re


class AICritic:
    """Вероятность, что русский текст написала LLM: toiletsandpaper/russian-ai-text-detector-*."""

    name = "onnx"

    def __init__(self, model_id: str = "toiletsandpaper/russian-ai-text-detector-bert",
                 threshold: float = 0.5) -> None:
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        self._np = np
        self.threshold = threshold
        path = snapshot_download(model_id, allow_patterns=["inference.json", "model.onnx*", "tokenizer.json"])
        with open(f"{path}/inference.json", encoding="utf-8") as fh:
            self.spec = json.load(fh)
        self._session = ort.InferenceSession(f"{path}/{self.spec['file']}", providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(f"{path}/{self.spec['tokenizer']}")
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def score(self, text: str) -> float:
        """Вероятность ИИ-происхождения текста (0..1)."""
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

    def accept(self, text: str) -> bool:
        """True, если детектор больше не помечает текст как ИИ."""
        return self.score(text) < self.threshold
