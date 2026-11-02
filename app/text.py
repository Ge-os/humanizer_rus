"""Токенизация, стоп-слова и морфология русского текста."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pymorphy3
import razdel
from nltk.corpus import stopwords

_WORD_RE = re.compile(r"^[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-]*$")
_POS_FOR_INFLECT = {"NOUN", "ADJF", "ADJS", "PRTF", "PRTS", "VERB", "INFN", "GRND", "NUMR", "NPRO"}


@dataclass(frozen=True)
class Token:
    """Токен текста с позициями и морфологической информацией."""

    text: str
    start: int
    end: int
    is_word: bool
    is_stop: bool
    lemma: str


class TextProcessor:
    """Токенизация (razdel), стоп-слова (NLTK), лемматизация и склонение (pymorphy3)."""

    def __init__(self) -> None:
        self._morph = pymorphy3.MorphAnalyzer()
        stops = set(stopwords.words("russian"))
        self._stops = stops | {w.replace("ё", "е") for w in stops}
        self._parse_cache: dict[str, list[Token]] = {}
        self._lemma_cache: dict[str, str] = {}
        self._inflect_cache: dict[tuple[str, str], str] = {}

    def parse(self, text: str) -> list[Token]:
        """Разбирает текст в токены с признаками слова, стоп-слова и леммы (с кэшем)."""
        cached = self._parse_cache.get(text)
        if cached is not None:
            return cached
        tokens: list[Token] = []
        for item in razdel.tokenize(text):
            raw = item.text
            is_word = bool(_WORD_RE.match(raw))
            norm = raw.lower().replace("ё", "е")
            lemma = self.lemma(raw) if is_word else ""
            tokens.append(
                Token(raw, item.start, item.stop, is_word, is_word and norm in self._stops, lemma)
            )
        self._parse_cache[text] = tokens
        return tokens

    def lemma(self, word: str) -> str:
        """Нормальная форма слова (с кэшем)."""
        key = word.lower()
        cached = self._lemma_cache.get(key)
        if cached is None:
            cached = self._morph.parse(key)[0].normal_form
            self._lemma_cache[key] = cached
        return cached

    def target_indices(self, tokens: list[Token], n: int) -> list[int]:
        """Индексы каждого n-го значимого токена (не стоп-слова)."""
        out: list[int] = []
        seen = 0
        for i, tok in enumerate(tokens):
            if not tok.is_word or tok.is_stop:
                continue
            seen += 1
            if seen % n == 0:
                out.append(i)
        return out

    def inflect(self, word: str, reference: str) -> str:
        """Приводит word к грамматической форме reference (род/число/падеж/одушевлённость)."""
        if not word or " " in word or not _WORD_RE.match(word):
            return self.match_case(word, reference)
        key = (word.lower(), reference.lower())
        cached = self._inflect_cache.get(key)
        if cached is None:
            cached = self._inflect_uncached(word.lower(), reference.lower())
            self._inflect_cache[key] = cached
        return self.match_case(cached, reference)

    def _inflect_uncached(self, word: str, reference: str) -> str:
        """Возвращает словоформу без учёта регистра (склонение/спряжение)."""
        ref = self._morph.parse(reference)[0]
        if ref.tag.POS not in _POS_FOR_INFLECT:
            return word
        grams = {getattr(ref.tag, g) for g in ("case", "number") if getattr(ref.tag, g, None)}
        if getattr(ref.tag, "case", None) == "accs" and ref.tag.animacy:
            grams.add(ref.tag.animacy)
        if getattr(ref.tag, "number", None) == "sing" and ref.tag.gender:
            grams.add(ref.tag.gender)
        if ref.tag.POS in ("VERB", "INFN", "GRND"):
            grams |= {getattr(ref.tag, g) for g in ("person", "tense", "mood") if getattr(ref.tag, g, None)}
        parses = sorted(self._morph.parse(word), key=lambda p: 0 if p.tag.POS == ref.tag.POS else 1)
        for parsed in parses:
            if parsed.tag.POS not in _POS_FOR_INFLECT:
                continue
            inflected = parsed.inflect(grams)
            if inflected is not None:
                return inflected.word
        return word

    @staticmethod
    def match_case(word: str, reference: str) -> str:
        """Копирует регистр reference на word; для строчного reference снимает заглавную."""
        if reference.isupper():
            return word.upper()
        if reference[:1].isupper():
            return word[:1].upper() + word[1:]
        return word[:1].lower() + word[1:]

    def window(self, tokens: list[Token], index: int, size: int) -> tuple[str, str]:
        """Возвращает левый и правый контекст токена в пределах size слов с каждой стороны."""
        left_words: list[str] = []
        for tok in reversed(tokens[:index]):
            if tok.is_word:
                left_words.append(tok.text)
                if len(left_words) >= size:
                    break
        right_words: list[str] = []
        for tok in tokens[index + 1 :]:
            if tok.is_word:
                right_words.append(tok.text)
                if len(right_words) >= size:
                    break
        return " ".join(reversed(left_words)), " ".join(right_words)

    def sentence_bounds(self, text: str, start: int, end: int) -> tuple[str, str]:
        """Возвращает левую и правую части текста в границах предложения вокруг позиции."""
        stops = ".!?…\n"
        left = start
        while left > 0 and text[left - 1] not in stops:
            left -= 1
        right = end
        while right < len(text) and text[right] not in stops:
            right += 1
        return text[left:start], text[end:right]
