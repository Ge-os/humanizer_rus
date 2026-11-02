"""Подготовка данных без тяжёлых загрузок: ИИ-тексты LLMTrace и малый корпус.

Используется datasets-server API HuggingFace: строки берутся выборками,
без скачивания всего датасета (экономия мобильного трафика).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

BASE = "https://datasets-server.huggingface.co"
DATASET = "iitolstykh/LLMTrace_classification"


@dataclass
class Sample:
    """Выбранный текст с метаданными."""

    text_id: str
    text: str
    label: str
    model: str
    data_type: str


class LLMTraceClient:
    """Клиент datasets-server: постраничная выборка строк LLMTrace."""

    def __init__(self, dataset: str = DATASET) -> None:
        self.dataset = dataset

    def rows(self, split: str, offset: int, length: int = 100) -> list[dict]:
        """Возвращает до length строк split начиная с offset (с повтором при 429)."""
        for attempt in range(6):
            resp = requests.get(
                f"{BASE}/rows",
                params={"dataset": self.dataset, "config": "default", "split": split,
                        "offset": offset, "length": length},
                timeout=120,
            )
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            time.sleep(1.5)
            return [r["row"] for r in resp.json()["rows"]]
        raise RuntimeError(f"rate limited at offset {offset}")

    def collect(self, split: str, label: str, limit: int, min_len: int = 200,
                max_len: int = 10000, lang: str = "ru") -> list[Sample]:
        """Собирает до limit строк заданного языка и метки в диапазоне длин."""
        out: list[Sample] = []
        offset = 0
        while len(out) < limit:
            batch = self.rows(split, offset, 100)
            if not batch:
                break
            for row in batch:
                text = row.get("text") or ""
                if row.get("lang") != lang or row.get("label") != label:
                    continue
                if not (min_len <= len(text) <= max_len):
                    continue
                out.append(Sample(f"{split}_{offset}", text, label, row.get("model", ""), row.get("data_type", "")))
                if len(out) >= limit:
                    break
            offset += 100
        return out


class DataPrep:
    """Сохраняет ИИ-тексты для проверки и малый корпус для обучения моделей."""

    def __init__(self, out_dir: str | Path = "research/data") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.client = LLMTraceClient()

    @staticmethod
    def to_sentences(text: str) -> list[str]:
        """Грубое разбиение текста на предложения для корпуса."""
        return [s.strip() for s in re.split(r"[.!?…]+", text) if len(s.strip().split()) >= 3]

    def build(self, n_ai: int = 30, n_human: int = 800) -> None:
        """Собирает ai_texts.jsonl и corpus.txt в out_dir (существующее не перекачивается)."""
        ai_path = self.out_dir / "ai_texts.jsonl"
        have_ai = len(ai_path.read_text(encoding="utf-8").splitlines()) if ai_path.exists() else 0
        if have_ai >= n_ai:
            print(f"ai texts: {have_ai} already in {ai_path}")
        else:
            ai = self.client.collect("test", "ai", n_ai, min_len=400, max_len=2500)
            with open(ai_path, "w", encoding="utf-8") as fh:
                for i, s in enumerate(ai):
                    fh.write(json.dumps({"id": f"ai_{i:03d}", "text": s.text, "model": s.model,
                                         "data_type": s.data_type}, ensure_ascii=False) + "\n")
            print(f"ai texts: {len(ai)} -> {ai_path}")
        corpus_path = self.out_dir / "corpus.txt"
        human = self.client.collect("train", "human", n_human, min_len=200, max_len=5000)
        with open(corpus_path, "w", encoding="utf-8") as fh:
            for s in human:
                for sent in self.to_sentences(s.text):
                    fh.write(sent + "\n")
        print(f"corpus lines: {sum(1 for _ in open(corpus_path, encoding='utf-8'))} -> {corpus_path}")


if __name__ == "__main__":
    DataPrep().build()
