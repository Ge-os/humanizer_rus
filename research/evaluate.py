"""Матрица прогонов гуманизатора: тексты, логи решений и метрики сохраняются в research/results.

Запуск из корня репозитория: python research/evaluate.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.humanizer import Humanizer  # noqa: E402
from app.policy import ReplacementPolicy  # noqa: E402
from app.rankers import build_ranker  # noqa: E402
from app.synonyms import SynonymProvider  # noqa: E402
from app.text import TextProcessor  # noqa: E402

RESULTS = ROOT / "research" / "results"
DATA = ROOT / "research" / "data"


@dataclass(frozen=True)
class Config:
    """Конфигурация прогона гуманизатора."""

    model: str
    n: int
    mode: int
    threshold: float

    @property
    def tag(self) -> str:
        return f"{self.model}_n{self.n}_m{self.mode}_t{self.threshold}"


class Evaluator:
    """Гоняет конфигурации по текстам и пишет результаты на диск."""

    def __init__(self, samples: Path, corpus: Path, synonyms: Path = ROOT / "data" / "synonims.parquet") -> None:
        self.samples = [json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines() if line]
        self.corpus = corpus
        self.processor = TextProcessor()
        self.provider = SynonymProvider(synonyms)
        self.rankers = {}
        self.detectors = {}
        self._before: dict[tuple[str, str], float] = {}
        from detectors import PerplexityDetector

        self.detectors["ppl"] = PerplexityDetector(corpus)
        try:
            from detectors import OnnxDetector

            self.detectors["onnx"] = OnnxDetector()
        except Exception as exc:  # noqa: BLE001 - детектор опционален
            print(f"onnx detector skipped: {exc}", flush=True)

    def _score_before(self, detector, name: str, text_id: str, text: str) -> float:
        key = (name, text_id)
        if key not in self._before:
            self._before[key] = detector.score(text)
        return self._before[key]

    def _ranker(self, model: str):
        if model not in self.rankers:
            self.rankers[model] = build_ranker(model, models_dir=ROOT / "models", corpus=self.corpus)
        return self.rankers[model]

    def run(self, configs: list[Config]) -> list[dict]:
        """Прогоняет все конфигурации, возвращает и сохраняет сводные метрики."""
        rows: list[dict] = []
        for cfg in configs:
            humanizer = Humanizer(
                self.processor, self.provider, self._ranker(cfg.model),
                ReplacementPolicy(threshold=cfg.threshold, mode=cfg.mode, seed=42),
                n=cfg.n, context="sentence",
            )
            out_dir = RESULTS / "humanized" / cfg.tag
            log_dir = RESULTS / "logs" / cfg.tag
            out_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            for sample in self.samples:
                t0 = time.time()
                result = humanizer.humanize(sample["text"])
                row = {
                    "config": cfg.tag, "text_id": sample["id"], "model": cfg.model,
                    "n": cfg.n, "mode": cfg.mode, "threshold": cfg.threshold,
                    "decisions": len(result.decisions),
                    "replaced": sum(d.replaced for d in result.decisions),
                    "sec": round(time.time() - t0, 2),
                }
                if row["decisions"]:
                    row["replaced_share"] = round(row["replaced"] / row["decisions"], 3)
                else:
                    row["replaced_share"] = 0.0
                for name, detector in self.detectors.items():
                    before = self._score_before(detector, name, sample["id"], sample["text"])
                    after = detector.score(result.text)
                    row[f"{name}_before"] = round(before, 3)
                    row[f"{name}_after"] = round(after, 3)
                    row[f"{name}_delta"] = round(after - before, 3)
                (out_dir / f"{sample['id']}.txt").write_text(result.text, encoding="utf-8")
                (log_dir / f"{sample['id']}.json").write_text(
                    json.dumps({"config": cfg.tag, "text_id": sample["id"], "decisions": [
                        {"index": d.index, "original": d.original, "chosen": d.chosen, "replaced": d.replaced,
                         "candidates": [{"word": w, "prob": round(p, 4)} for w, p in d.candidates]}
                        for d in result.decisions]}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                rows.append(row)
                print(f"  {sample['id']}: {row['sec']}s замен {row['replaced']}/{row['decisions']}", flush=True)
            print(f"{cfg.tag}: done (всего замен {sum(r['replaced'] for r in rows if r['config'] == cfg.tag)})", flush=True)
            self._save(rows)
        return rows

    def _save(self, rows: list[dict]) -> None:
        """Сохраняет summary.csv и печатает агрегат по конфигурациям."""
        RESULTS.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "config", k != "text_id", k))
        with open(RESULTS / "summary.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        agg: dict[str, list[dict]] = {}
        for r in rows:
            agg.setdefault(r["config"], []).append(r)
        lines = ["| config | замен | ppl до | ppl после | Δppl | onnx до | onnx после | Δonnx |",
                 "|---|---|---|---|---|---|---|---|"]
        for tag, group in agg.items():
            def mean(key: str) -> float:
                vals = [g[key] for g in group if key in g and g[key] is not None]
                return sum(vals) / len(vals) if vals else float("nan")
            has_onnx = "onnx_before" in group[0]
            line = (f"| {tag} | {mean('replaced_share'):.2f} | {mean('ppl_before'):.1f} | "
                    f"{mean('ppl_after'):.1f} | {mean('ppl_delta'):+.1f} |")
            if has_onnx:
                line += f" {mean('onnx_before'):.3f} | {mean('onnx_after'):.3f} | {mean('onnx_delta'):+.3f} |"
            else:
                line += " - | - | - |"
            lines.append(line)
        table = "\n".join(lines)
        (RESULTS / "matrix.md").write_text(table + "\n", encoding="utf-8")
        print(table)


def main() -> None:
    """Собирает конфигурации простых методов и запускает матрицу."""
    import argparse

    p = argparse.ArgumentParser(description="Матрица прогонов гуманизатора")
    p.add_argument("--limit-texts", type=int, default=0, help="ограничить число текстов (0 = все)")
    p.add_argument("--models", default="lesk,ngram,pmi")
    p.add_argument("--ns", default="1,2,3")
    p.add_argument("--thresholds", default="0.25,0.5")
    args = p.parse_args()
    models = args.models.split(",")
    ns = [int(x) for x in args.ns.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]
    configs = [Config(model, n, 2, th) for model in models for n in ns for th in thresholds]
    configs.append(Config(models[0], ns[min(1, len(ns) - 1)], 1, thresholds[0]))
    ev = Evaluator(DATA / "ai_texts.jsonl", DATA / "corpus.txt")
    if args.limit_texts:
        ev.samples = ev.samples[: args.limit_texts]
    ev.run(configs)


if __name__ == "__main__":
    main()
