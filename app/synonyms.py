"""Словарь синонимов из synonims.parquet (колонки word, synonyms)."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd
import pymorphy3


class SynonymProvider:
    """Индекс по слову и по лемме: word -> [синонимы].

    Лемматизированный индекс строится один раз и кэшируется рядом с parquet.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._morph = pymorphy3.MorphAnalyzer()
        self._by_word: dict[str, list[str]] = {}
        self._by_lemma: dict[str, list[str]] = {}
        self._lookup_cache: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        df = pd.read_parquet(self._path)
        cache = self._path.with_name(f".{self._path.stem}_index.pkl")
        if cache.exists() and cache.stat().st_mtime >= self._path.stat().st_mtime:
            with open(cache, "rb") as fh:
                self._by_word, self._by_lemma = pickle.load(fh)
            return
        morph = pymorphy3.MorphAnalyzer()
        t0 = time.time()
        for word, syns in zip(df["word"], df["synonyms"]):
            key = word.lower().replace("ё", "е")
            clean = [s for s in syns if s and s.lower().replace("ё", "е") != key]
            if not clean:
                continue
            bucket = self._by_word.setdefault(key, [])
            for s in clean:
                if s not in bucket:
                    bucket.append(s)
            if " " not in word:
                lemma = morph.parse(word.lower())[0].normal_form.replace("ё", "е")
                lbucket = self._by_lemma.setdefault(lemma, [])
                for s in clean:
                    if s not in lbucket:
                        lbucket.append(s)
        with open(cache, "wb") as fh:
            pickle.dump((self._by_word, self._by_lemma), fh)
        print(f"[SynonymProvider] index built in {time.time() - t0:.0f}s, cached to {cache.name}")

    def lookup(self, word: str) -> list[str]:
        """Возвращает синонимы слова: точное совпадение, затем совпадение по лемме (с кэшем)."""
        key = word.lower().replace("ё", "е")
        cached = self._lookup_cache.get(key)
        if cached is not None:
            return list(cached)
        found = self._by_word.get(key)
        if not found:
            lemma = self._morph.parse(key)[0].normal_form.replace("ё", "е")
            found = self._by_lemma.get(lemma, [])
        self._lookup_cache[key] = list(found)
        return list(found)

    def __len__(self) -> int:
        return len(self._by_word)
