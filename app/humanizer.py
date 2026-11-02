"""Оркестрация: выбор n-го токена, поиск синонимов, ранжирование и замена (или омоглифы)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .homoglyphs import HomoglyphMapper
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
    """Заменяет каждый n-й значимый токен текста на подходящий синоним или омоглиф.

    Режимы 1–3: список [оригинал, синонимы...] ранжируется моделью в контексте,
    политика решает, заменять ли слово. Режим 4: буквы токена меняются на визуальные
    двойники из другого алфавита (омоглифы), словарь и модель не нужны.
    """

    def __init__(
        self,
        processor: TextProcessor,
        provider: SynonymProvider | None,
        ranker: BaseRanker | None,
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
        if policy.mode != 4 and (provider is None or ranker is None):
            raise ValueError("synonym modes require provider and ranker")
        self.processor = processor
        self.provider = provider
        self.ranker = ranker
        self.policy = policy
        self.n = n
        self.context = context
        self.window = window
        self.max_candidates = max_candidates
        self.mapper = HomoglyphMapper()

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

    def _homoglyph(self, index: int, tokens) -> Decision | None:
        """Режим 4: переводит токен на буквы другого алфавита (омоглифы)."""
        original = tokens[index].text
        if not self.mapper.changed(original):
            return None
        chosen = self.mapper.convert(original)
        return Decision(index, original, chosen, True, [(original, 1.0), (chosen, 1.0)])

    def humanize(self, text: str) -> HumanizeResult:
        """Гуманизирует текст, сохраняя структуру и регистр."""
        tokens = self.processor.parse(text)
        decisions: list[Decision] = []
        replacements: list[tuple[int, int, str]] = []
        for index in self.processor.target_indices(tokens, self.n):
            if self.policy.mode == 4:
                decision = self._homoglyph(index, tokens)
            else:
                decision = self._synonym(index, text, tokens)
            if decision is None:
                continue
            decisions.append(decision)
            if decision.replaced:
                replacements.append((tokens[index].start, tokens[index].end, decision.chosen))
        out = text
        for start, end, word in reversed(replacements):
            out = out[:start] + word + out[end:]
        return HumanizeResult(out, decisions)

    def _synonym(self, index: int, text: str, tokens) -> Decision | None:
        """Режимы 1–3: ранжирование синонимов и выбор по политике."""
        original = tokens[index].text
        candidates = self._candidates(original)
        if len(candidates) < 2:
            return None
        left, right = self._context(text, tokens, index)
        ranked = self.ranker.rank(left, right, candidates)
        chosen = self.policy.pick(ranked, original)
        return Decision(index, original, chosen, chosen != original, ranked)
