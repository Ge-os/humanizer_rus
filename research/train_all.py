"""Обучение всех артефактов моделей на корпусе и сохранение в models/.

Запуск из корня: python research/train_all.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rankers import REGISTRY, build_ranker  # noqa: E402
from app.synonyms import SynonymProvider  # noqa: E402


def main() -> None:
    """Тренирует каждый needs_corpus-ранжировщик и пишет models/<name>.pkl."""
    corpus = ROOT / "research" / "data" / "corpus.txt"
    provider = SynonymProvider(ROOT / "data" / "synonims.parquet")
    for name, cls in REGISTRY.items():
        if not cls.needs_corpus:
            continue
        t0 = time.time()
        build_ranker(name, models_dir=ROOT / "models", corpus=corpus, synonyms=provider)
        print(f"{name}: trained in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
