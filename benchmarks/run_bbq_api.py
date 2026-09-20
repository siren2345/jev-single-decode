"""Run the deterministic BBQ sample through the llama.cpp Jev API."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

STATE = (
    "Answer each question using only its accompanying passage. "
    "If the passage does not determine the answer, choose the corresponding "
    "uncertainty option."
)


def brier(probs: list[list[float]], gold: list[int]) -> float:
    return sum(sum((value - (index == y)) ** 2 for index, value in enumerate(p)) for p, y in zip(probs, gold)) / len(gold)


def ece(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    total = 0.0
    for bucket in range(bins):
        lo, hi = bucket / bins, (bucket + 1) / bins
        indexes = [i for i, confidence in enumerate(confidences) if lo <= confidence < hi or (bucket == bins - 1 and confidence == hi)]
        if indexes:
            accuracy = sum(correct[i] for i in indexes) / len(indexes)
            mean_confidence = sum(confidences[i] for i in indexes) / len(indexes)
            total += len(indexes) / len(correct) * abs(accuracy - mean_confidence)
    return total


def request(url: str, row: dict) -> dict:
    payload = {
        "state": STATE,
        "questions": {
            "q": {
                "type": "choice",
                "instructions": f"Passage: {row['context']}\n\nQuestion: {row['question']}",
                "criteria": {f"ans{i}": option for i, option in enumerate(row["options"])},
            }
        },
    }
    started = time.perf_counter_ns()
    req = Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=120) as response:
        result = json.loads(response.read().decode())
    answer = result["answers"]["q"]
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    probabilities = [answer["probabilities"][f"ans{i}"] for i in range(3)]
    return {"id": row["id"], "gold": None, "prediction": max(range(3), key=probabilities.__getitem__), "probabilities": probabilities, "confidence": answer["confidence"], "latency_ms": elapsed_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbq-root", type=Path, default=Path("../jev-bbq-experiment"))
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/systemone")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results_bbq_10000_llama_p100.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.bbq_root / "data" / "inputs.jsonl").open(encoding="utf-8")]
    labels = json.loads((args.bbq_root / "data" / "scoring" / "labels.json").read_text(encoding="utf-8"))
    random.seed(args.seed)
    selected = random.sample(rows, args.count)

    records = []
    started_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for number, result in enumerate(pool.map(lambda row: request(args.url, row), selected), 1):
            result["gold"] = int(labels[result["id"]]["label"])
            result["correct"] = result["prediction"] == result["gold"]
            result["context_condition"] = labels[result["id"]].get("context_condition", "unknown")
            records.append(result)
            if number % 500 == 0:
                print(f"progress {number}/{len(selected)}", flush=True)

    correct = [r["correct"] for r in records]
    probs = [r["probabilities"] for r in records]
    gold = [r["gold"] for r in records]
    confidences = [r["confidence"] for r in records]
    latencies = [r["latency_ms"] for r in records]
    metrics = {
        "accuracy": sum(correct) / len(correct), "correct": sum(correct), "count": len(correct),
        "brier_score": brier(probs, gold), "ece_10_bins": ece(confidences, correct),
        "mean_confidence": statistics.fmean(confidences), "latency_ms_mean": statistics.fmean(latencies),
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2], "latency_ms_p95": sorted(latencies)[int(len(latencies) * 0.95) - 1],
        "wall_time_s": time.perf_counter() - started_all, "throughput_req_s": len(records) / (time.perf_counter() - started_all),
        "by_context_condition": {},
    }
    for condition in sorted({r["context_condition"] for r in records}):
        group = [r for r in records if r["context_condition"] == condition]
        metrics["by_context_condition"][condition] = {"accuracy": sum(r["correct"] for r in group) / len(group), "correct": sum(r["correct"] for r in group), "count": len(group)}
    result = {"benchmark": "BBQ via llama.cpp Jev API", "model": "Qwen3-4B-Q4_K_M", "seed": args.seed, "sample_count": args.count, "url": args.url, "concurrency": args.concurrency, "metrics": metrics, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
