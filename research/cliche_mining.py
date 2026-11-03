"""Добыча RU AI-клише на расширенном пуле LLMTrace: фразы, перепредставленные в ИИ-текстах.

Фильтр — только «функциональные» рамки (глаголы-связки, вводные конструкции), доменные
термины отбрасываются. Кэш пула: research/data/llmtrace_pool.jsonl.
Запуск из корня: python research/cliche_mining.py [--ai 800] [--human 800]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research" / "data"

FUNCTION_STARTERS = (
    "является", "являться", "представляет", "позволяет", "способствует", "требует", "играет",
    "служит", "выступает", "помогает", "обеспечивает", "заключается", "составляет", "определяется",
    "основан", "направлен", "связан", "находится", "обладает", "предоставляет", "создаёт",
    "формирует", "важно", "необходимо", "следует", "стоит", "можно", "как", "таким", "кроме",
    "более", "прежде", "поэтому", "однако", "при этом", "вместе", "наряду", "по мере", "в целях",
    "с точки зрения", "в контексте", "важность", "значение", "роль", "процесс", "аспект",
)


SHARD_URL = ("https://huggingface.co/api/datasets/iitolstykh/LLMTrace_classification/"
             "parquet/default/train/0.parquet")


def fetch_pool(n_ai: int, n_human: int) -> list[dict]:
    """Собирает пул из parquet-шарда LLMTrace (потоково, кэш research/data/llmtrace_pool.jsonl)."""
    cache = DATA / "llmtrace_pool.jsonl"
    rows: list[dict] = []
    if cache.exists():
        rows = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l]
        if sum(1 for r in rows if r["label"] == "ai") >= n_ai and \
           sum(1 for r in rows if r["label"] == "human") >= n_human:
            return rows
    shard = DATA / "llmtrace_train0.parquet"
    if not shard.exists():
        import requests

        with requests.get(SHARD_URL, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(shard, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
        print(f"shard downloaded: {shard.stat().st_size / 1e6:.1f} MB", flush=True)
    import pyarrow.parquet as pq

    got = {"ai": sum(1 for r in rows if r["label"] == "ai"),
           "human": sum(1 for r in rows if r["label"] == "human")}
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=200, columns=["lang", "label", "text"]):
        data = batch.to_pydict()
        for lang, label, text in zip(data["lang"], data["label"], data["text"]):
            if lang != "ru" or len(text) < 150 or len(text) > 6000 or got.get(label, n_ai) >= (n_ai if label == "ai" else n_human):
                continue
            rows.append({"label": label, "text": text})
            got[label] = got.get(label, 0) + 1
        if got.get("ai", 0) >= n_ai and got.get("human", 0) >= n_human:
            break
    cache.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"pool: ai={got.get('ai', 0)} human={got.get('human', 0)}", flush=True)
    return rows


def mine(rows: list[dict], top: int = 60) -> list[tuple[float, int, str]]:
    """Считает log-odds фраз ai против human и фильтрует по функциональным рамкам."""
    ai_grams: Counter = Counter()
    human_grams: Counter = Counter()
    for r in rows:
        import re

        tokens = re.findall(r"[а-яё]+", r["text"].lower())
        target = ai_grams if r["label"] == "ai" else human_grams
        for n in (2, 3, 4):
            for i in range(len(tokens) - n + 1):
                target[" ".join(tokens[i : i + n])] += 1
    ai_total = sum(ai_grams.values()) or 1
    human_total = sum(human_grams.values()) or 1
    scored = []
    for gram, c in ai_grams.items():
        if c < 5:
            continue
        if not any(gram.startswith(s) or f" {s}" in f" {gram}" for s in FUNCTION_STARTERS):
            continue
        p_ai = (c + 0.5) / ai_total
        p_h = (human_grams.get(gram, 0) + 0.5) / human_total
        score = math.log(p_ai / p_h)
        if score > math.log(3):
            scored.append((score, c, gram))
    scored.sort(reverse=True)
    return scored[:top]


def main() -> None:
    ap = argparse.ArgumentParser(description="Добыча RU AI-клише")
    ap.add_argument("--ai", type=int, default=800)
    ap.add_argument("--human", type=int, default=800)
    args = ap.parse_args()
    rows = fetch_pool(args.ai, args.human)
    for score, c, gram in mine(rows):
        print(f"{score:5.2f} ai={c:4d}  {gram}")


if __name__ == "__main__":
    main()
