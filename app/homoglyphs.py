"""Замена букв на визуально неотличимые из другого алфавита (омоглифы): a <-> а, o <-> о."""
from __future__ import annotations

_PAIRS = (
    ("a", "а"), ("c", "с"), ("e", "е"), ("o", "о"), ("p", "р"), ("x", "х"), ("y", "у"),
    ("i", "і"), ("j", "ј"), ("s", "ѕ"),
    ("A", "А"), ("B", "В"), ("C", "С"), ("E", "Е"), ("H", "Н"), ("K", "К"), ("M", "М"),
    ("O", "О"), ("P", "Р"), ("T", "Т"), ("X", "Х"), ("Y", "У"),
)

SWAP: dict[str, str] = {lat: cyr for lat, cyr in _PAIRS}
SWAP.update({cyr: lat for lat, cyr in _PAIRS})


class HomoglyphMapper:
    """Инвертирует алфавит букв-омоглифов в слове: каждая буква заменяется на двойник.

    Работает в обе стороны (латиница -> кириллица и обратно), регистр сохраняется.
    """

    def convert(self, word: str) -> str:
        """Возвращает слово с буквами из другого алфавита (визуально без изменений)."""
        return "".join(SWAP.get(ch, ch) for ch in word)

    def changed(self, word: str) -> bool:
        """Есть ли в слове буквы, доступные для замены."""
        return any(ch in SWAP for ch in word)
