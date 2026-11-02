"""CLI-запуск гуманизатора: файл на вход, файл на выход, параметры флагами."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ai_transforms import (BackTranslationTransformer, ClicheTransformer,
                            PolishTransformer, RhythmTransformer)
from .humanizer import Humanizer
from .policy import ReplacementPolicy
from .rankers import REGISTRY, build_ranker
from .synonyms import SynonymProvider
from .text import TextProcessor

AI_STEPS = {"rhythm", "cliche", "polish", "homoglyph", "synonym", "backtranslate"}


def build_parser() -> argparse.ArgumentParser:
    """Собирает парсер аргументов командной строки."""
    p = argparse.ArgumentParser(prog="humanizer", description="Гуманизация русского AI-текста заменой n-го токена на синоним")
    p.add_argument("--input", "-i", required=True, help="входной файл с текстом (utf-8)")
    p.add_argument("--output", "-o", required=True, help="выходной файл с результатом")
    p.add_argument("--synonyms", default="data/synonims.parquet", help="parquet со словарём синонимов")
    p.add_argument("--model", default="ngram", choices=sorted(REGISTRY), help="модель ранжирования кандидатов")
    p.add_argument("--n", type=int, default=1, help="каждый n-й значимый токен (не стоп-слово)")
    p.add_argument("--mode", type=int, default=2, choices=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                   help="1–3 — выбор синонима (случайный/вероятный/наименее вероятный), "
                        "4 — омоглифы, 5 — ритм предложений (LLM), 6 — замена AI-клише (LLM), "
                        "7 — полировка (LLM), 8 — цепочка AI-шагов из --chain, "
                        "9 — back-translation (opus-mt или LLM)")
    p.add_argument("--chain", default="cliche,rhythm,polish",
                   help="шаги режима 8 через запятую: rhythm,cliche,polish,homoglyph,synonym,backtranslate")
    p.add_argument("--critic", default="none", choices=("none", "onnx"),
                   help="детектор-критик для полировки (режимы 7/8): onnx — итеративно правит, "
                        "пока детектор не перестанет помечать текст как ИИ")
    p.add_argument("--threshold", type=float, default=0.5, help="порог вероятности для замены")
    p.add_argument("--seed", type=int, default=None, help="seed для режима 1")
    p.add_argument("--context", default="sentence", choices=("sentence", "window"), help="контекст для ранжирования")
    p.add_argument("--window", type=int, default=10, help="размер окна (слов) при --context window")
    p.add_argument("--max-candidates", type=int, default=10, help="максимум кандидатов с оригиналом")
    p.add_argument("--corpus", default=None, help="корпус (предложение в строке) для обучения ngram/pmi")
    p.add_argument("--models-dir", default="models", help="каталог с артефактами моделей")
    p.add_argument("--report", default=None, help="куда сохранить JSON-отчёт с решениями")
    return p


def _build_ai_transform(args: argparse.Namespace):
    """Собирает LLM-трансформацию по режиму 5–8 (цепочка может включать homoglyph/synonym)."""
    from .ai_transforms import (BackTranslationTransformer, ChainTransformer, ClicheTransformer,
                                HumanizerStep, PolishTransformer, RhythmTransformer)
    from .llm import LLMClient

    client = LLMClient()
    processor = TextProcessor()
    critic = None
    if args.critic == "onnx":
        from .critic import AICritic

        critic = AICritic()

    def humanizer_step(name: str):
        if name == "homoglyph":
            policy = ReplacementPolicy(threshold=args.threshold, mode=4, seed=args.seed)
            humanizer = Humanizer(processor, None, None, policy, n=args.n)
        else:
            provider = SynonymProvider(args.synonyms)
            ranker = build_ranker(args.model, models_dir=args.models_dir, corpus=args.corpus, synonyms=provider)
            policy = ReplacementPolicy(threshold=args.threshold, mode=2, seed=args.seed)
            humanizer = Humanizer(processor, provider, ranker, policy, n=args.n)
        return HumanizerStep(name, humanizer)

    makers = {
        "rhythm": lambda: RhythmTransformer(client, seed=args.seed),
        "cliche": lambda: ClicheTransformer(client, seed=args.seed),
        "polish": lambda: PolishTransformer(client, seed=args.seed, critic=critic, max_iter=3),
        "homoglyph": lambda: humanizer_step("homoglyph"),
        "synonym": lambda: humanizer_step("synonym"),
        "backtranslate": lambda: BackTranslationTransformer(client, seed=args.seed),
    }
    if args.mode == 5:
        return makers["rhythm"]()
    if args.mode == 6:
        return makers["cliche"]()
    if args.mode == 7:
        return makers["polish"]()
    if args.mode == 9:
        return makers["backtranslate"]()
    names = [s.strip() for s in args.chain.split(",") if s.strip()]
    unknown = [s for s in names if s not in AI_STEPS]
    if unknown or not names:
        raise SystemExit(f"--chain: неизвестные шаги {unknown}, доступны {sorted(AI_STEPS)}")
    return ChainTransformer(client, [makers[s]() for s in names], seed=args.seed)


def main(argv: list[str] | None = None) -> None:
    """Читает файл, гуманизирует, пишет результат и опциональный отчёт."""
    args = build_parser().parse_args(argv)
    text = Path(args.input).read_text(encoding="utf-8")
    if args.mode >= 5:
        transform = _build_ai_transform(args)
        result_text = transform.transform(text)
        steps = [s.name for s in transform.steps] if hasattr(transform, "steps") else [transform.name]
        print(f"mode={args.mode} steps={','.join(steps)}")
        report_body = {"steps": steps}
    else:
        processor = TextProcessor()
        policy = ReplacementPolicy(threshold=args.threshold, mode=args.mode, seed=args.seed)
        if args.mode == 4:
            provider, ranker = None, None
        else:
            provider = SynonymProvider(args.synonyms)
            ranker = build_ranker(args.model, models_dir=args.models_dir, corpus=args.corpus, synonyms=provider)
        humanizer = Humanizer(
            processor, provider, ranker, policy,
            n=args.n, context=args.context, window=args.window, max_candidates=args.max_candidates,
        )
        result = humanizer.humanize(text)
        result_text = result.text
        replaced = sum(d.replaced for d in result.decisions)
        print(f"model={args.model} n={args.n} mode={args.mode} threshold={args.threshold}")
        print(f"candidates={len(result.decisions)} replaced={replaced}")
        report_body = {
            "decisions": [
                {"index": d.index, "original": d.original, "chosen": d.chosen, "replaced": d.replaced,
                 "candidates": [{"word": w, "prob": p} for w, p in d.candidates]}
                for d in result.decisions
            ],
        }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result_text, encoding="utf-8")
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"params": vars(args), **report_body}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
