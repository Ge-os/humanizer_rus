"""AI-трансформации текста через LLM: ритм предложений, замена клише, полировка.

Все трансформации просят LLM возвращать строго полезную нагрузку (предложение
или JSON со списком вариантов) — без метатекста. Ошибки парсинга не роняют
прогон: фрагмент остаётся неизменным.
"""
from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod

from .llm import LLMClient

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_CYR = re.compile(r"[а-яёА-ЯЁ]")

CLICHES_RU = (
    "в современном мире", "на сегодняшний день", "в настоящее время", "в мире, где",
    "играет важную роль", "играет ключевую роль", "играет важнейшую роль", "служит основой",
    "лежит в основе", "имеет важное значение", "имеет большое значение", "является важным",
    "является ключевым", "является одним из", "представляет собой", "представляет важность",
    "следует отметить", "необходимо отметить", "стоит отметить", "важно отметить", "важно отметить, что",
    "необходимо учитывать", "важно учитывать", "следует учитывать", "как уже упоминалось",
    "как известно", "прежде всего", "в заключение следует", "в заключение стоит", "в заключение необходимо",
    "подводя итог", "итоге следует сказать", "таким образом можно сделать вывод",
    "широко используется", "широкий спектр", "ряд преимуществ", "ряд факторов", "ряд возможностей",
    "уникальная возможность", "неоспоримый факт", "динамично развивается", "постоянно развивается",
    "на данном этапе", "осуществляется", "данный момент", "как правило", "в ряде случаев",
    "в свою очередь", "более того", "кроме того", "вместе с тем", "наряду с этим", "в целом",
    "позволяет значительно", "способствует развитию", "требует от специалистов", "век технологий",
    "с каждым днём", "каждый день всё", "не только", "так или иначе", "по сути дела",
    "это позволяет", "что позволяет", "что способствует", "важно отметить, что", "важно отметить что",
    "важно помнить", "важно помнить, что", "очень важно", "подчеркивает важность",
    "стоит обратить внимание", "стоит обратить внимание на", "также стоит", "однако стоит",
    "однако несмотря на", "такие как", "таких как", "составляет около", "можно увидеть",
    "не является", "направлены на", "является важным", "ключевую роль", "важно чтобы",
)
CLICHES_EN = (
    "delve into", "tapestry of", "in today's world", "it is important to note",
    "it is worth noting", "plays a crucial role", "plays a vital role", "in conclusion",
    "landscape of", "realm of", "navigate the", "a testament to", "pivotal", "showcasing",
    "multifaceted", "holistic approach", "deep dive", "game-changer", "in the ever-evolving",
    "leverage", "when it comes to", "serves as a", "stands as a", "not only", "in light of",
    "it's crucial to", "boasts a", "rich tapestry", "a myriad of", "furthermore",
)


def is_russian(text: str) -> bool:
    """Определяет русский текст по доле кириллицы."""
    return len(_CYR.findall(text)) >= len(text) / 20


def split_sentences(text: str) -> list[str]:
    """Грубое разбиение на предложения с сохранением разделителей."""
    parts = _SENT_SPLIT.split(text.strip())
    return [p for p in parts if p.strip()]


class BaseAITransform(ABC):
    """Базовая LLM-трансформация текста."""

    name = "base"

    def __init__(self, client: LLMClient, seed: int | None = None) -> None:
        self.client = client
        self._rng = random.Random(seed)

    @abstractmethod
    def transform(self, text: str) -> str:
        """Возвращает преобразованный текст."""

    def _variants(self, prompt: str, system: str | None = None) -> list[str]:
        """Вызов LLM с ожиданием JSON {"variants": [...]}; fallback — сырой текст."""
        try:
            data = self.client.complete_json(prompt, system)
            out = [v.strip() for v in data.get("variants", []) if isinstance(v, str) and v.strip()]
            if out:
                return out
        except (ValueError, RuntimeError):
            pass
        try:
            raw = self.client.complete(prompt, system).strip()
        except RuntimeError:
            return []
        return [raw] if raw else []

    def _pick(self, variants: list[str]) -> str | None:
        return self._rng.choice(variants) if variants else None

    @staticmethod
    def _keep_punct(original: str, chosen: str | None) -> str | None:
        """Сохраняет конечную пунктуацию исходного предложения."""
        if chosen is None:
            return None
        tail = original.rstrip()[-1:]
        chosen = chosen.strip()
        if tail in ".!?…" and chosen and chosen[-1] not in ".!?…":
            chosen += tail
        return chosen


class RhythmTransformer(BaseAITransform):
    """Режим 5: разрушение моноритма — длинные предложения дробит/сжимает, короткие расширяет."""

    name = "rhythm"

    def __init__(self, client: LLMClient, seed: int | None = None,
                 long_words: int = 22, short_words: int = 7) -> None:
        super().__init__(client, seed)
        self.long_words = long_words
        self.short_words = short_words

    def _prompt(self, sentence: str, kind: str, ru: bool) -> str:
        if kind == "split":
            task = ("Предложение слишком длинное и однообразное. Разбей его на 2-3 коротких "
                    "ИЛИ сократи, не добавляя новой информации и не меняя факты и смысл."
                    if ru else
                    "The sentence is too long and monotonous. Split it into 2-3 short ones "
                    "OR shorten it, adding no new information and keeping facts and meaning.")
        else:
            task = ("Предложение слишком короткое и рвёт ритм текста. Расширь его до естественной "
                    "длины, добавляя только уточнения из контекста БЕЗ новой информации и фактов."
                    if ru else
                    "The sentence is too short and breaks the rhythm. Expand it to a natural "
                    "length using only contextual details, adding NO new information or facts.")
        how = ("Верни СТРОГО JSON вида {\"variants\": [\"вариант 1\", \"вариант 2\"]} — минимум "
               "2 варианта, без какого-либо текста вне JSON." if ru else
               "Return STRICTLY JSON like {\"variants\": [\"variant 1\", \"variant 2\"]} — "
               "at least 2 variants, no text outside JSON.")
        return f"{task} {how}\n{'Предложение' if ru else 'Sentence'}: {sentence}"

    def transform(self, text: str) -> str:
        ru = is_russian(text)
        out = []
        for sent in split_sentences(text):
            n_words = len(sent.split())
            kind = "split" if n_words >= self.long_words else ("expand" if n_words <= self.short_words else None)
            chosen = None
            if kind:
                chosen = self._keep_punct(sent, self._pick(self._variants(self._prompt(sent, kind, ru))))
            out.append(chosen or sent)
        sep = " " if ru else " "
        return sep.join(out)


class ClicheTransformer(BaseAITransform):
    """Режим 6: находит AI-клише в предложениях и минимально их переформулирует."""

    name = "cliche"

    def __init__(self, client: LLMClient, seed: int | None = None) -> None:
        super().__init__(client, seed)
        self.cliches_ru = CLICHES_RU
        self.cliches_en = CLICHES_EN

    def _find(self, sentence: str, ru: bool) -> list[str]:
        low = sentence.lower()
        pool = self.cliches_ru if ru else self.cliches_en
        return [c for c in pool if c in low]

    def _prompt(self, sentence: str, found: list[str], ru: bool) -> str:
        items = ", ".join(f"«{c}»" for c in found)
        if ru:
            return (f"В предложении есть штампы AI-текста: {items}. Переформулируй предложение, "
                    "заменив штампы МИНИМАЛЬНО (минимум слов), сохранив смысл, факты и числа. "
                    "Верни СТРОГО предложение без кавычек и пояснений.\n"
                    f"Предложение: {sentence}")
        return (f"The sentence contains AI clichés: {items}. Rephrase it replacing the clichés "
                "MINIMALLY (minimum words), keeping meaning, facts and numbers. "
                "Return STRICTLY the sentence, no quotes or explanations.\n"
                f"Sentence: {sentence}")

    def transform(self, text: str) -> str:
        ru = is_russian(text)
        out = []
        for sent in split_sentences(text):
            found = self._find(sent, ru)
            chosen = None
            if found:
                chosen = self._keep_punct(sent, self._pick(self._variants(self._prompt(sent, found, ru))))
            out.append(chosen or sent)
        return " ".join(out)


class PolishTransformer(BaseAITransform):
    """Режим 7: мягкая академическая полировка с гейтом метрики и опциональным критиком.

    critic(after_text) -> True, если текст принят (например, детектор больше не помечает
    его как ИИ); при отказе модель правит ещё раз, до max_iter итераций.
    """

    name = "polish"

    def __init__(self, client: LLMClient, seed: int | None = None,
                 gate=None, chunk_chars: int = 3000, critic=None, max_iter: int = 3) -> None:
        super().__init__(client, seed)
        self.gate = gate  # Callable[[str, str], bool]: (до, после) -> принять?
        self.critic = critic
        self.max_iter = max_iter
        self.chunk_chars = chunk_chars

    def _prompt(self, chunk: str, ru: bool) -> str:
        if ru:
            return ("Перепиши текст так, чтобы он стал более академичным и походил на научную "
                    "статью. ТЕБЕ НЕ ЗАПРЕЩЕНО менять целые предложения, но исправляй только "
                    "неуместные/некорректные/неакадемичные обороты, заменяя МИНИМУМ слов. Цель — "
                    "оставить текст прежним, исправив лишь конструкции, не соответствующие "
                    "академическому стилю, сделать его структурированным, грамотным по пунктуации "
                    "и грамматике и приятным для чтения. Не меняй факты и существенную информацию. "
                    "Идеи, значение и семантическое содержание должны остаться прежними. "
                    "Верни ТОЛЬКО итоговый текст полностью:\n\n" + chunk)
        return ("Rewrite the text so that it becomes more academic and resembles a paper. "
                "YOU ARE NOT FORBIDDEN TO CHANGE WHOLE SENTENCES, correct only inappropriate / "
                "incorrect / unacademic sentences or turns, replacing MINIMUM words. Your goal "
                "is to leave the text the same, correcting only constructions that do not conform "
                "to the academic writing style, to make the text more structured, punctuationally "
                "and grammatically correct and pleasant to read. Do not change facts or essential "
                "information. Ideas, meaning and semantic content should remain the same. "
                "The text should be finalised and returned in full:\n\n" + chunk)

    @staticmethod
    def _chunks(text: str, limit: int) -> list[str]:
        out, buf = [], ""
        for para in re.split(r"(\n\s*\n)", text):
            if buf and len(buf) + len(para) > limit:
                out.append(buf)
                buf = para
            else:
                buf += para
        if buf.strip():
            out.append(buf)
        return out

    def _retry_prompt(self, polished: str, ru: bool) -> str:
        if ru:
            return ("Текст всё ещё выглядит сгенерированным ИИ. Перепиши СИЛЬНЕЕ: измени порядок "
                    "слов, длину и ритм предложений, убери штампы и канцелярит — но сохрани смысл, "
                    "факты и числа. Верни ТОЛЬКО итоговый текст полностью:\n\n" + polished)
        return ("The text still looks AI-generated. Rewrite it HARDER: change word order, sentence "
                "length and rhythm, remove clichés — but keep meaning, facts and numbers. "
                "Return STRICTLY the final text in full:\n\n" + polished)

    def transform(self, text: str) -> str:
        ru = is_russian(text)
        out = []
        for chunk in self._chunks(text, self.chunk_chars):
            polished = self.client.complete(self._prompt(chunk, ru)).strip()
            for _ in range(self.max_iter - 1):
                if not self.critic or self.critic(polished):
                    break
                polished = self.client.complete(self._retry_prompt(polished, ru)).strip()
            if self.gate and not self.gate(chunk, polished):
                polished = chunk
            out.append(polished)
        return "\n".join(out)


class HumanizerStep:
    """Адаптер механической стадии (омоглифы/синонимы) для AI-цепочки (режим 8)."""

    def __init__(self, name: str, humanizer) -> None:
        self.name = name
        self._humanizer = humanizer

    def transform(self, text: str) -> str:
        return self._humanizer.humanize(text).text


class BackTranslationTransformer(BaseAITransform):
    """Режим 9 / шаг backtranslate: перевод туда-обратно (RU→EN→RU) как ломка статистики.

    Использует Helsinki-NLP/opus-mt (transformers, ленивая загрузка); если модели или
    зависимости недоступны — back-translation через LLM-промпт.
    """

    name = "backtranslate"

    def __init__(self, client: LLMClient, seed: int | None = None,
                 max_chars: int = 1500, use_llm: bool = False) -> None:
        super().__init__(client, seed)
        self.max_chars = max_chars
        self._use_llm = use_llm
        self._models = None

    def _load(self) -> None:
        if self._models is not None or self._use_llm:
            return
        try:
            from transformers import MarianMTModel, MarianTokenizer

            models = {}
            for tag in ("ru-en", "en-ru"):
                name = f"Helsinki-NLP/opus-mt-{tag}"
                models[tag] = (MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name))
            self._models = models
        except Exception:  # noqa: BLE001 - нет torch/моделей → LLM-fallback
            self._use_llm = True

    def _translate(self, tag: str, text: str) -> str:
        import torch

        tokenizer, model = self._models[tag]
        torch.manual_seed(self._rng.randrange(2**31))
        batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=400)
        with torch.no_grad():
            out = model.generate(**batch, num_beams=1, do_sample=True, temperature=0.9,
                                 max_new_tokens=400)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    def _llm_bt(self, chunk: str, ru: bool) -> str:
        if ru:
            prompt = ("Переведи текст на английский язык, а затем обратно на русский — как "
                      "естественный человеческий перевод, слегка меняя порядок слов и ритм. "
                      "Сохрани смысл, факты и числа. Верни ТОЛЬКО итоговый русский текст:\n\n" + chunk)
        else:
            prompt = ("Translate the text into Russian and then back into English — as natural "
                      "human translation, slightly changing word order and rhythm. Keep meaning, "
                      "facts and numbers. Return STRICTLY the final English text:\n\n" + chunk)
        return self.client.complete(prompt).strip()

    def transform(self, text: str) -> str:
        self._load()
        ru = is_russian(text)
        out = []
        for sent in split_sentences(text):
            if self._use_llm:
                chosen = self._llm_bt(sent, ru) if len(sent) <= self.max_chars else sent
            else:
                mid = self._translate("ru-en", sent) if ru else self._translate("en-ru", sent)
                chosen = self._translate("en-ru", mid) if ru else self._translate("ru-en", mid)
            out.append(self._keep_punct(sent, chosen) or sent)
        return " ".join(out)


class ChainTransformer(BaseAITransform):
    """Режим 8: цепочка трансформаций, например cliche -> rhythm -> polish -> homoglyph."""

    name = "chain"

    def __init__(self, client: LLMClient, steps: list, seed: int | None = None) -> None:
        super().__init__(client, seed)
        self.steps = steps

    def transform(self, text: str) -> str:
        for step in self.steps:
            text = step.transform(text)
        return text
