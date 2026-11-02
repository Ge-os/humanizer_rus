"""Ранжировщики кандидатов-синонимов по левому и правому контексту.

Каждый ранжировщик получает left/right текста и список кандидатов
[оригинал, синоним 1, ...] и возвращает вероятности (softmax по скорам).
Модель выбирается одним флагом --model через REGISTRY.
"""
from __future__ import annotations

import math
import pickle
import re
import zipfile
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import nltk.data
import pymorphy3
import razdel
from nltk.lm import Vocabulary, WittenBellInterpolated
from nltk.util import everygrams

_WORD_RE = re.compile(r"^[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-]*$")


class RussianWordNetIndex:
    """Индекс русского WordNet (OMW: cldr + wikt): слово <-> синсеты PWN.

    В NLTK-пакетах omw нет русского, поэтому табы читаются из extended_omw напрямую.
    """

    def __init__(self) -> None:
        self.word_to_syn: dict[str, set[str]] = defaultdict(set)
        self.syn_to_words: dict[str, set[str]] = defaultdict(set)
        self._load()

    def _load(self) -> None:
        try:
            zpath = str(nltk.data.find("corpora/extended_omw.zip"))
        except LookupError:
            return
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if "rus" not in name or not name.endswith(".tab"):
                    continue
                with zf.open(name) as raw:
                    for line in raw:
                        line = line.decode("utf-8", "replace")
                        if line.startswith("#"):
                            continue
                        parts = line.strip().split("\t")
                        if len(parts) < 3 or "lemma" not in parts[1]:
                            continue
                        syn, word = parts[0], parts[2].strip().lower().replace("ё", "е")
                        self.word_to_syn[word].add(syn)
                        self.syn_to_words[syn].add(word)

    def lemmas_of(self, word: str) -> set[str]:
        """Все русские слова из синсетов слова (включая само слово)."""
        out = {word}
        for syn in self.word_to_syn.get(word, ()):
            out |= self.syn_to_words[syn]
        return out

    def __len__(self) -> int:
        return len(self.word_to_syn)


def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Преобразует скоры в вероятности (сумма 1); -inf трактуется как очень низкий скор."""
    if not scores:
        return []
    scaled = [s / temperature if math.isfinite(s) else -1e18 for s in scores]
    top = max(scaled)
    exps = [math.exp(s - top) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _words(text: str) -> list[str]:
    """Словесные токены нижнего регистра из строки контекста."""
    return [t.text.lower().replace("ё", "е") for t in razdel.tokenize(text) if _WORD_RE.match(t.text)]


def safe_logscore(model, word: str, context: tuple[str, ...]) -> float:
    """logscore с деградацией контекста и полом: nltk.lm KneserNey падает на <UNK>/<s> контекстах."""
    ctx = tuple(context)
    while True:
        try:
            return max(float(model.logscore(word, ctx)), -50.0)
        except ZeroDivisionError:
            if not ctx:
                return -50.0
            ctx = ctx[1:]


def _corpus_sentences(path: str | Path) -> list[list[str]]:
    """Корпус: предложение в строке -> списки слов."""
    sentences: list[list[str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            words = _words(line)
            if len(words) >= 2:
                sentences.append(words)
    return sentences


def train_ngram(sentences: list[list[str]], order: int = 3):
    """Обучает n-граммную модель: редкие слова (count < 2) заменяются на <UNK>.

    WittenBellInterpolated вместо KneserNey: последний в nltk.lm сканирует все
    n-граммы на каждый вызов и падает на незнакомых контекстах.
    """
    counts: dict[str, int] = {}
    for words in sentences:
        for w in words:
            counts[w] = counts.get(w, 0) + 1
    prepared = [["<UNK>" if counts[w] < 2 else w for w in s] for s in sentences]
    vocab = Vocabulary([w for s in prepared for w in s] + ["<s>"])
    model = WittenBellInterpolated(order, vocabulary=vocab)
    model.fit([list(everygrams(s, max_len=order)) for s in prepared])
    return model


class BaseRanker(ABC):
    """Базовый интерфейс ранжировщика кандидатов."""

    name = "base"
    needs_corpus = False

    @abstractmethod
    def score(self, left: str, right: str, candidates: list[str]) -> list[float]:
        """Сырые скоры кандидатов в паре (left, right)."""

    def rank(self, left: str, right: str, candidates: list[str]) -> list[tuple[str, float]]:
        """Ранжированный список (кандидат, вероятность) по убыванию."""
        probs = softmax(self.score(left, right, candidates))
        return sorted(zip(candidates, probs), key=lambda x: -x[1])


class LeskRanker(BaseRanker):
    """Lesk-подобный скор через русский WordNet: пересечение синсетов кандидата и контекста."""

    name = "lesk"

    def __init__(self, **_: object) -> None:
        self._wn = RussianWordNetIndex()
        self._morph = pymorphy3.MorphAnalyzer()
        self._lemma_cache: dict[str, str] = {}

    def _lemma(self, word: str) -> str:
        cached = self._lemma_cache.get(word)
        if cached is None:
            cached = self._morph.parse(word.lower())[0].normal_form.replace("ё", "е")
            self._lemma_cache[word] = cached
        return cached

    def score(self, left: str, right: str, candidates: list[str]) -> list[float]:
        ctx = {self._lemma(w) for w in _words(left) + _words(right)}
        out = []
        for cand in candidates:
            senses = self._wn.lemmas_of(self._lemma(cand)) if " " not in cand else set()
            out.append(float(len(senses & ctx)))
        return out


class NgramRanker(BaseRanker):
    """N-граммная языковая модель (Kneser-Ney через nltk.lm): log P(кандидат | левый) + правая часть."""

    name = "ngram"
    needs_corpus = True

    def __init__(self, order: int = 3, model_path: str | Path | None = None, corpus: str | Path | None = None, **_: object) -> None:
        self.order = order
        self._model: KneserNeyInterpolated | None = None
        if model_path and Path(model_path).exists():
            self.load(model_path)
        elif corpus:
            self.fit(_corpus_sentences(corpus))
            if model_path:
                self.save(model_path)

    def fit(self, sentences: list[list[str]]) -> None:
        """Обучает n-граммную модель на предложениях (списки слов)."""
        self._model = train_ngram(sentences, self.order)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"order": self.order, "model": self._model}, fh)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.order, self._model = data["order"], data["model"]

    def _score_one(self, left_words: list[str], right_words: list[str], cand: str) -> float:
        pad = self.order - 1
        ctx = (["<s>"] * pad + left_words[-pad:])[-pad:] if pad else []
        total = safe_logscore(self._model, cand, tuple(ctx))
        trail = list(ctx) + [cand]
        for word in right_words[:pad]:
            total += safe_logscore(self._model, word, tuple(trail[-pad:]))
            trail.append(word)
        return total

    def score(self, left: str, right: str, candidates: list[str]) -> list[float]:
        left_words, right_words = _words(left), _words(right)
        return [self._score_one(left_words, right_words, c.lower().replace("ё", "е")) for c in candidates]


class PMIRanker(BaseRanker):
    """Точечная взаимная информация: средняя PMI кандидата со словами контекста."""

    name = "pmi"
    needs_corpus = True

    def __init__(self, window: int = 5, model_path: str | Path | None = None, corpus: str | Path | None = None, **_: object) -> None:
        self.window = window
        self._uni: dict[str, int] = {}
        self._pair: dict[tuple[str, str], int] = {}
        self._total = 0
        if model_path and Path(model_path).exists():
            self.load(model_path)
        elif corpus:
            self.fit(_corpus_sentences(corpus))
            if model_path:
                self.save(model_path)

    def fit(self, sentences: list[list[str]]) -> None:
        """Считает частоты слов и совместную встречаемость в окне window."""
        for words in sentences:
            for i, w in enumerate(words):
                self._uni[w] = self._uni.get(w, 0) + 1
                self._total += 1
                for j in range(i + 1, min(i + 1 + self.window, len(words))):
                    key = (w, words[j]) if w <= words[j] else (words[j], w)
                    self._pair[key] = self._pair.get(key, 0) + 1

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"window": self.window, "uni": self._uni, "pair": self._pair, "total": self._total}, fh)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.window, self._uni, self._pair, self._total = data["window"], data["uni"], data["pair"], data["total"]

    def _pmi(self, a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        cab = self._pair.get(key, 0)
        ca, cb = self._uni.get(a, 0), self._uni.get(b, 0)
        return math.log((cab + 1) * self._total / ((ca + 1) * (cb + 1)))

    def score(self, left: str, right: str, candidates: list[str]) -> list[float]:
        ctx = (_words(left) + _words(right))[-2 * self.window :]
        out = []
        for cand in candidates:
            word = cand.lower().replace("ё", "е")
            out.append(sum(self._pmi(word, w) for w in ctx) / max(len(ctx), 1))
        return out


class SklearnRankerBase(BaseRanker):
    """Классификатор «слово подходит контексту» на BoW(left + кандидат + right).

    Обучение: настоящие слова корпуса (метка 1) против случайных слов (метка 0).
    Скор — логит «подходит»; rank() переводит его в вероятности через softmax.
    """

    needs_corpus = True

    def __init__(self, model_path: str | Path | None = None, corpus: str | Path | None = None,
                 synonyms=None, window: int = 3, max_features: int = 5000,
                 max_positions: int = 30000, seed: int = 42, **_: object) -> None:
        self.window = window
        self.max_features = max_features
        self.max_positions = max_positions
        self.seed = seed
        self._vectorizer = None
        self._clf = None
        if model_path and Path(model_path).exists():
            self.load(model_path)
        elif corpus:
            self.fit(_corpus_sentences(corpus), synonyms)
            if model_path:
                self.save(model_path)

    def _doc(self, left: str, right: str, cand: str) -> str:
        return " ".join(_words(left)[-self.window :] + [cand.lower().replace("ё", "е")] + _words(right)[: self.window])

    def _make_clf(self):  # noqa: ANN201 - тип зависит от подкласса
        raise NotImplementedError

    def _fit_clf(self, x, labels) -> None:  # noqa: ANN001
        self._clf = self._make_clf()
        self._clf.fit(x, labels)

    def _features(self, x):  # noqa: ANN001, ANN201
        return x

    def _margin(self, proba) -> float:  # noqa: ANN001
        """Логит класса 1 из predict_proba (переопределяется для линейных моделей)."""
        import math

        p1, p0 = max(float(proba[1]), 1e-9), max(float(proba[0]), 1e-9)
        return math.log(p1) - math.log(p0)

    def fit(self, sentences: list[list[str]], synonyms=None) -> None:
        """Учит TF-IDF + классификатор на парах (контекст, слово) -> подходит/нет."""
        import random as _random

        from sklearn.feature_extraction.text import TfidfVectorizer

        rng = _random.Random(self.seed)
        vocab = sorted({w for s in sentences for w in s if len(w) > 2})
        docs: list[str] = []
        labels: list[int] = []
        positions = [(s, i) for s in sentences for i, w in enumerate(s) if len(w) > 2]
        rng.shuffle(positions)
        for words, i in positions[: self.max_positions]:
            left = " ".join(words[max(0, i - self.window) : i])
            right = " ".join(words[i + 1 : i + 1 + self.window])
            docs.append(self._doc(left, right, words[i]))
            labels.append(1)
            for _ in range(2):
                docs.append(self._doc(left, right, rng.choice(vocab)))
                labels.append(0)
        self._vectorizer = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2))
        x = self._vectorizer.fit_transform(docs)
        self._fit_clf(x, labels)

    def save(self, path: str | Path) -> None:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump({"vectorizer": self._vectorizer, "clf": self._clf,
                         "extra": self._extra_state(),
                         "window": self.window, "max_features": self.max_features}, fh)

    def _extra_state(self):  # noqa: ANN201
        return None

    def _load_extra(self, state) -> None:  # noqa: ANN001
        pass

    def load(self, path: str | Path) -> None:
        import pickle

        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._vectorizer, self._clf = data["vectorizer"], data["clf"]
        self._load_extra(data.get("extra"))
        self.window, self.max_features = data["window"], data["max_features"]

    def score(self, left: str, right: str, candidates: list[str]) -> list[float]:
        docs = [self._doc(left, right, c) for c in candidates]
        x = self._features(self._vectorizer.transform(docs))
        if hasattr(self._clf, "predict_proba"):
            return [self._margin(p) for p in self._clf.predict_proba(x)]
        return [float(v) for v in self._clf.decision_function(x)]


class NbRanker(SklearnRankerBase):
    """Наивный Байес (MultinomialNB) над BoW окна."""

    name = "nb"

    def _make_clf(self):
        from sklearn.naive_bayes import MultinomialNB

        return MultinomialNB(alpha=0.1)


class LogregRanker(SklearnRankerBase):
    """Логистическая регрессия над BoW окна."""

    name = "logreg"

    def _make_clf(self):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, C=1.0)


class SvmRanker(SklearnRankerBase):
    """LinearSVC над BoK окна (скор — decision_function)."""

    name = "svm"

    def _make_clf(self):
        from sklearn.svm import LinearSVC

        return LinearSVC(C=0.5, max_iter=2000)


class RfRanker(SklearnRankerBase):
    """Random Forest над BoW окна (ограниченная глубина — иначе модель не сериализуется)."""

    name = "rf"

    def _make_clf(self):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=self.seed)


class GbmRanker(SklearnRankerBase):
    """Gradient Boosting (HistGradientBoosting) поверх SVD-признаков окна."""

    name = "gbm"

    def _make_clf(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=200, random_state=self.seed)

    def _fit_clf(self, x, labels) -> None:  # noqa: ANN001
        from sklearn.decomposition import TruncatedSVD

        self._svd = TruncatedSVD(n_components=128, random_state=self.seed)
        self._clf = self._make_clf()
        self._clf.fit(self._svd.fit_transform(x), labels)

    def _features(self, x):  # noqa: ANN001, ANN201
        return self._svd.transform(x)

    def _extra_state(self):  # noqa: ANN201
        return self._svd

    def _load_extra(self, state) -> None:  # noqa: ANN001
        self._svd = state


class LsaRanker(BaseRanker):
    """LSA: косинус между вектором кандидата и центроидом контекста в TF-IDF+SVD пространстве."""

    name = "lsa"
    needs_corpus = True

    def __init__(self, model_path: str | Path | None = None, corpus: str | Path | None = None,
                 n_components: int = 128, **_: object) -> None:
        self._tfidf = None
        self._svd = None
        if model_path and Path(model_path).exists():
            self.load(model_path)
        elif corpus:
            self.fit(_corpus_sentences(corpus))
            if model_path:
                self.save(model_path)

    def fit(self, sentences: list[list[str]]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self._normalize = normalize
        self._tfidf = TfidfVectorizer(max_features=10000)
        x = self._tfidf.fit_transform(" ".join(s) for s in sentences)
        self._svd = TruncatedSVD(n_components=128, random_state=42)
        self._svd.fit(x)

    def save(self, path: str | Path) -> None:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump({"tfidf": self._tfidf, "svd": self._svd}, fh)

    def load(self, path: str | Path) -> None:
        import pickle

        from sklearn.preprocessing import normalize

        self._normalize = normalize
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._tfidf, self._svd = data["tfidf"], data["svd"]

    def _vec(self, text: str):
        return self._svd.transform(self._tfidf.transform([text]))

    def score(self, left: str, right: str, candidates: list[str]) -> list[float]:
        import numpy as np

        ctx_words = _words(left)[-6:] + _words(right)[:6]
        ctx = self._normalize(self._vec(" ".join(ctx_words)))
        out = []
        for cand in candidates:
            vec = self._normalize(self._vec(cand.lower().replace("ё", "е")))
            out.append(float(np.dot(ctx, vec.T)[0, 0]))
        return out


REGISTRY: dict[str, type[BaseRanker]] = {
    LeskRanker.name: LeskRanker,
    NgramRanker.name: NgramRanker,
    PMIRanker.name: PMIRanker,
    NbRanker.name: NbRanker,
    LogregRanker.name: LogregRanker,
    SvmRanker.name: SvmRanker,
    RfRanker.name: RfRanker,
    GbmRanker.name: GbmRanker,
    LsaRanker.name: LsaRanker,
}


def build_ranker(name: str, models_dir: str | Path = "models", corpus: str | Path | None = None, **kwargs) -> BaseRanker:
    """Создаёт ранжировщик по имени: загружает артефакт или обучает на корпусе."""
    if name not in REGISTRY:
        raise ValueError(f"unknown model '{name}', available: {sorted(REGISTRY)}")
    cls = REGISTRY[name]
    if cls.needs_corpus:
        artifact = Path(models_dir) / f"{name}.pkl"
        return cls(model_path=artifact, corpus=corpus, **kwargs)  # type: ignore[call-arg]
    return cls(**kwargs)  # type: ignore[call-arg]
