"""Проверка сохранённых гуманизированных текстов ONNX-детектором.

Считает скор ИИ для всех humanized/<config>/ai_*.txt и сравнивает с базовой линией
(оригиналы в detector_baseline.csv). Запуск из корня: python research/detector_check.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from detectors import OnnxDetector  # noqa: E402

RESULTS = ROOT / "research" / "results"


def main() -> None:
    """Считает onnx-скоры всех сохранённых вариантов и печатает сводку по конфигурациям."""
    det = OnnxDetector()
    base = {}
    with open(RESULTS / "detector_baseline.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["label"] == "ai":
                base[row["text_id"]] = float(row["onnx"])
    rows = []
    for cfg_dir in sorted((RESULTS / "humanized").iterdir()):
        if not cfg_dir.is_dir():
            continue
        scores, flips, deltas = 0, 0, []
        for f in sorted(cfg_dir.glob("ai_*.txt")):
            score = det.score(f.read_text(encoding="utf-8"))
            before = base[f.stem]
            rows.append({"config": cfg_dir.name, "text_id": f.stem,
                         "onnx_before": round(before, 4), "onnx_after": round(score, 4),
                         "delta": round(score - before, 4), "flipped": int(before >= 0.5 > score)})
            scores += 1
            flips += int(before >= 0.5 > score)
            deltas.append(score - before)
        print(f"{cfg_dir.name}: текстов {scores}, снято с детекции {flips}, "
              f"Δonnx среднее {sum(deltas) / max(len(deltas), 1):+.4f}, "
              f"мин {min(deltas, default=0):+.4f}", flush=True)
    with open(RESULTS / "detector_scores.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["config", "text_id", "onnx_before", "onnx_after", "delta", "flipped"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved research/results/detector_scores.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
