"""Run the canonical 100-question BBQ benchmark locally.

Usage:
  python benchmarks/run_bbq_100.py --bbq-root ../jev-bbq-experiment
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_single_decode import LocalChoiceJudge, STATE  # noqa: E402


def load_rows(root: Path):
    rows = [json.loads(line) for line in (root / "data" / "inputs.jsonl").open(encoding="utf-8")]
    labels = json.loads((root / "data" / "scoring" / "labels.json").read_text(encoding="utf-8"))
    return rows, labels


def brier(probs: list[list[float]], gold: list[int]) -> float:
    total = 0.0
    for p, y in zip(probs, gold):
        total += sum((value - (index == y)) ** 2 for index, value in enumerate(p))
    return total / len(gold)


def ece(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    total = 0.0
    n = len(correct)
    for bucket in range(bins):
        lo = bucket / bins
        hi = (bucket + 1) / bins
        indexes = [
            i for i, confidence in enumerate(confidences)
            if lo <= confidence < hi or (bucket == bins - 1 and confidence == hi)
        ]
        if indexes:
            accuracy = sum(correct[i] for i in indexes) / len(indexes)
            mean_confidence = sum(confidences[i] for i in indexes) / len(indexes)
            total += len(indexes) / n * abs(accuracy - mean_confidence)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbq-root", type=Path, default=Path("jev-bbq-experiment"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results_bbq_100.json")
    args = parser.parse_args()

    rows, labels = load_rows(args.bbq_root)
    random.seed(args.seed)
    selected = random.sample(rows, args.count)
    judge = LocalChoiceJudge(args.model)

    records = []
    for number, row in enumerate(selected, 1):
        gold = int(labels[row["id"]]["label"])
        condition = labels[row["id"]].get("context_condition", "unknown")
        started = time.perf_counter()
        choice, probabilities, confidence = judge.choice_distribution(
            row, state=STATE
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        records.append({
            "id": row["id"],
            "gold": gold,
            "prediction": choice,
            "probabilities": probabilities,
            "confidence": confidence,
            "correct": choice == gold,
            "context_condition": condition,
            "latency_ms": elapsed_ms,
        })
        if number % 10 == 0:
            print(f"progress {number}/{len(selected)}", flush=True)

    correct = [record["correct"] for record in records]
    probabilities = [record["probabilities"] for record in records]
    gold = [record["gold"] for record in records]
    confidences = [record["confidence"] for record in records]
    latencies = [record["latency_ms"] for record in records]

    def subset(condition: str):
        return [record for record in records if record["context_condition"] == condition]

    metrics = {
        "accuracy": sum(correct) / len(correct),
        "correct": sum(correct),
        "count": len(correct),
        "brier_score": brier(probabilities, gold),
        "ece_10_bins": ece(confidences, correct),
        "mean_confidence": sum(confidences) / len(confidences),
        "mean_confidence_correct": sum(r["confidence"] for r in records if r["correct"]) / max(sum(correct), 1),
        "mean_confidence_incorrect": sum(r["confidence"] for r in records if not r["correct"]) / max(len(records) - sum(correct), 1),
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2],
        "latency_ms_p95": sorted(latencies)[int(len(latencies) * 0.95) - 1],
        "by_context_condition": {},
    }
    for condition in sorted({r["context_condition"] for r in records}):
        group = subset(condition)
        metrics["by_context_condition"][condition] = {
            "accuracy": sum(r["correct"] for r in group) / len(group),
            "correct": sum(r["correct"] for r in group),
            "count": len(group),
        }

    result = {
        "benchmark": "BBQ via jev-bbq-experiment",
        "model": args.model,
        "seed": args.seed,
        "sample_count": args.count,
        "prompt": "Qwen chat template; original BBQ state; thinking disabled; one A/B/C decision",
        "metrics": metrics,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
