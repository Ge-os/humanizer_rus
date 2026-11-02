"""Пилот AI-режимов (5–8) на нескольких ИИ-текстах: метрики onnx/ppl + сохранение текстов.

Запуск из корня: python research/ai_pilot.py [--limit 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from app.ai_transforms import ChainTransformer, ClicheTransformer, PolishTransformer, RhythmTransformer  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from detectors import OnnxDetector, PerplexityDetector  # noqa: E402

RESULTS = ROOT / "research" / "results" / "ai_mode"
DATA = ROOT / "research" / "data"


def main() -> None:
    """Гоняет AI-трансформации по текстам и пишет ai_mode/ai_pilot.csv."""
    ap = argparse.ArgumentParser(description="Пилот AI-режимов")
    ap.add_argument("--limit", type=int, default=4, help="число текстов")
    ap.add_argument("--skip-onnx", action="store_true", help="без onnx-детектора")
    args = ap.parse_args()

    samples = [json.loads(l) for l in (DATA / "ai_texts.jsonl").read_text(encoding="utf-8").splitlines() if l][: args.limit]
    client = LLMClient()
    onnx = None if args.skip_onnx else OnnxDetector()
    ppl = PerplexityDetector(DATA / "corpus.txt")
    critic = (lambda t: onnx.score(t) < 0.5) if onnx else None

    def homoglyph_step():
        from app.ai_transforms import HumanizerStep
        from app.humanizer import Humanizer
        from app.policy import ReplacementPolicy
        from app.text import TextProcessor

        return HumanizerStep("homoglyph", Humanizer(TextProcessor(), None, None,
                                                    ReplacementPolicy(mode=4), n=2))

    configs = {
        "rhythm": lambda: RhythmTransformer(client, seed=42),
        "cliche": lambda: ClicheTransformer(client, seed=42),
        "polish": lambda: PolishTransformer(client, seed=42),
        "polish_critic": lambda: PolishTransformer(client, seed=42, critic=critic, max_iter=3),
        "chain_crp": lambda: ChainTransformer(client, [ClicheTransformer(client, seed=42),
                                                       RhythmTransformer(client, seed=42),
                                                       PolishTransformer(client, seed=42)], seed=42),
        "chain_ph": lambda: ChainTransformer(client, [PolishTransformer(client, seed=42),
                                                      homoglyph_step()], seed=42),
    }
    rows = []
    for name, make in configs.items():
        transform = make()
        out_dir = RESULTS / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in samples:
            t0 = time.time()
            out_text = transform.transform(s["text"])
            (out_dir / f"{s['id']}.txt").write_text(out_text, encoding="utf-8")
            row = {"config": name, "text_id": s["id"], "sec": round(time.time() - t0, 1),
                   "ppl_before": round(ppl.score(s["text"]), 1), "ppl_after": round(ppl.score(out_text), 1)}
            row["ppl_delta"] = round(row["ppl_after"] - row["ppl_before"], 1)
            if onnx:
                b, a = onnx.score(s["text"]), onnx.score(out_text)
                row.update(onnx_before=round(b, 4), onnx_after=round(a, 4), onnx_delta=round(a - b, 4),
                           flipped=int(b >= 0.5 > a))
            rows.append(row)
            print(f"{name}/{s['id']}: {row['sec']}s ppl {row['ppl_before']}->{row['ppl_after']} "
                  f"onnx {row.get('onnx_before')}-{row.get('onnx_after')}", flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / "ai_pilot.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("\n=== сводка (среднее) ===")
    for name in configs:
        g = [r for r in rows if r["config"] == name]
        flip = sum(r.get("flipped", 0) for r in g)
        print(f"{name}: Δppl {sum(r['ppl_delta'] for r in g) / len(g):+.0f} | "
              f"onnx {sum(r['onnx_before'] for r in g) / len(g):.3f} -> "
              f"{sum(r['onnx_after'] for r in g) / len(g):.3f} | снято {flip}/{len(g)}")


if __name__ == "__main__":
    main()
