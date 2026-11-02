"""Политика выбора замены: порог вероятности и режим выбора кандидата."""
from __future__ import annotations

import random


class ReplacementPolicy:
    """Отбирает кандидата из ранжированного списка [(слово, вероятность)].

    Режим 1 — случайный кандидат среди прошедших порог (кроме оригинала),
    режим 2 — самый вероятный прошедший порог (кроме оригинала),
    режим 3 — наименее вероятный прошедший порог (кроме оригинала): максимизирует
    сюрприз токена и потому повышает перплексию текста.
    Если прошёл порог только оригинал или никто — остаётся оригинал.
    """

    def __init__(self, threshold: float = 0.5, mode: int = 2, seed: int | None = None) -> None:
        if mode not in (1, 2, 3):
            raise ValueError("mode must be 1, 2 or 3")
        self.threshold = threshold
        self.mode = mode
        self._rng = random.Random(seed)

    def pick(self, ranked: list[tuple[str, float]], original: str) -> str:
        """Возвращает выбранное слово по правилам порога и режима."""
        passed = [(w, p) for w, p in ranked if p >= self.threshold and w != original]
        if not passed:
            return original
        if self.mode == 1:
            return self._rng.choice(passed)[0]
        if self.mode == 3:
            return min(passed, key=lambda wp: wp[1])[0]
        return max(passed, key=lambda wp: wp[1])[0]
