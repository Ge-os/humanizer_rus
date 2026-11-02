"""Оркестрация: выбор n-го токена, поиск синонимов, ранжирование и замена."""
from __future__ import annotations

from dataclasses import dataclass, field

from .policy import ReplacementPolicy
from .rankers import BaseRanker
from .synonyms import SynonymProvider
from .text import TextProcessor


@dataclass
class Decision:
    """Решение по одному токену: кандидаты, вероятности, итоговая замена."""

    index: int
    original: str
    chosen: str
    replaced: bool
    candidates: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class HumanizeResult:
    """Результат гуманизации текста и лог решений."""

    text: str
    decisions: list[Decision]


class Humanizer:
    """Заменяет каждый n-й значимый токен текста на подходящий синоним.

    Для токена строится список [оригинал, синонимы...], ранжировщик оценивает
    его в контексте, политика решает, заменять ли слово.
    """

    def __init__(
        self,
        processor: TextProcessor,
        provider: SynonymProvider,
        ranker: BaseRanker,
        policy: ReplacementPolicy,
        n: int = 1,
        context: str = "sentence",
        window: int = 10,
        max_candidates: int = 10,
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if context not in ("sentence", "window"):
            raise ValueError("context must be 'sentence' or 'window'")
        self.processor = processor
        self.provider = provider
        self.ranker = ranker
        self.policy = policy
        self.n = n
        self.context = context
        self.window = window
        self.max_candidates = max_candidates

    def _candidates(self, original: str) -> list[str]:
        """Кандидаты: оригинал + синонимы, склонённые под форму оригинала."""
        out = [original]
        for syn in self.provider.lookup(original):
            word = self.processor.inflect(syn, original)
            if word and word not in out:
                out.append(word)
            if len(out) >= self.max_candidates:
                break
        return out

    def _context(self, text: str, tokens, index: int) -> tuple[str, str]:
        if self.context == "window":
            return self.processor.window(tokens, index, self.window)
        tok = tokens[index]
        return self.processor.sentence_bounds(text, tok.start, tok.end)

    def humanize(self, text: str) -> HumanizeResult:
        """Гуманизирует текст, сохраняя структуру и регистр."""
        tokens = self.processor.parse(text)
        decisions: list[Decision] = []
        replacements: list[tuple[int, int, str]] = []
        for index in self.processor.target_indices(tokens, self.n):
            original = tokens[index].text
            candidates = self._candidates(original)
            if len(candidates) < 2:
                continue
            left, right = self._context(text, tokens, index)
            ranked = self.ranker.rank(left, right, candidates)
            chosen = self.policy.pick(ranked, original)
            replaced = chosen != original
            decisions.append(Decision(index, original, chosen, replaced, ranked))
            if replaced:
                replacements.append((tokens[index].start, tokens[index].end, chosen))
        out = text
        for start, end, word in reversed(replacements):
            out = out[:start] + word + out[end:]
        return HumanizeResult(out, decisions)
