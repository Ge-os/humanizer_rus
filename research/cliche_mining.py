"""Добыча RU AI-клише: фразы, перепредставленные в ИИ-текстах против human-корпуса.

Запуск из корня: python research/cliche_mining.py -> печатает кандидатов для CLICHES_RU.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "data"

_WORD = re.compile(r"[а-яё]+")
_STOP = {"в", "и", "что", "на", "с", "по", "для", "не", "как", "это", "к", "из", "о", "от", "до",
         "а", "но", "или", "же", "бы", "за", "при", "со", "у", "ее", "его", "их", "мы", "вы", "он",
         "она", "оно", "они", "я", "ты", "то", "так", "все", "также", "уже", "еще", "ее"}


def ngrams(tokens: list[str], n_min: int = 2, n_max: int = 4) -> Counter:
    """Считает n-граммы (2–4 слова), начинающиеся не со стоп-слова."""
    out = Counter()
    for n in range(n_min, n_max + 1):
        for i in range(len(tokens) - n + 1):
            gram = tokens[i : i + n]
            if gram[0] in _STOP or gram[-1] in _STOP:
                continue
            out[" ".join(gram)] += 1
    return out


def tokens_of(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def main() -> None:
    """Печатает топ-фразы, специфичные для ИИ-текстов."""
    ai_text = "\n".join(json.loads(l)["text"]
                        for l in (DATA / "ai_texts.jsonl").read_text(encoding="utf-8").splitlines() if l)
    human_text = (DATA / "corpus.txt").read_text(encoding="utf-8")
    ai_grams = ngrams(tokens_of(ai_text))
    human_grams = ngrams(tokens_of(human_text))
    ai_total = sum(ai_grams.values()) or 1
    human_total = sum(human_grams.values()) or 1
    scored = []
    for gram, c in ai_grams.items():
        if c < 3:
            continue
        p_ai = c / ai_total
        p_h = (human_grams.get(gram, 0) + 0.5) / human_total
        scored.append((p_ai / p_h, c, gram))
    scored.sort(reverse=True)
    for ratio, c, gram in scored[:40]:
        print(f"{ratio:6.1f}  ai={c:3d}  {gram}")


if __name__ == "__main__":
    main()
