"""Measure end-to-end latency of the llama.cpp-backed Jev API.

This deliberately measures the public adapter, including HTTP and JSON
overhead, so it can be compared with the latency reported by the local
Transformers implementation. Run it after llama-server and the adapter are
started. No third-party package is required.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def payload_for(index: int) -> dict:
    return {
        "state": "Answer each question using only its accompanying passage.",
        "questions": {
            "who": {
                "type": "choice",
                "instructions": (
                    f"Passage: Alice went home. Unique test passage {index}.\n\n"
                    "Question: Who went home?"
                ),
                "criteria": {"ans0": "Alice", "ans1": "Bob", "ans2": "Can't be determined"},
            }
        },
    }


def request(url: str, index: int) -> dict:
    body = json.dumps(payload_for(index), ensure_ascii=False).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def timed_request(url: str, index: int) -> float:
    started = time.perf_counter_ns()
    result = request(url, index)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    assert result["answers"]["who"]["choice"] == "ans0", result
    return elapsed_ms


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/jev")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(timed_request, [args.url] * args.warmup, range(args.warmup)))

    samples = []
    batch_latencies = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        remaining = args.count
        while remaining:
            batch_size = min(args.concurrency, remaining)
            started = time.perf_counter_ns()
            indexes = range(args.count - remaining, args.count - remaining + batch_size)
            samples.extend(pool.map(timed_request, [args.url] * batch_size, indexes))
            batch_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            remaining -= batch_size

    print(json.dumps({
        "url": args.url,
        "count": len(samples),
        "concurrency": args.concurrency,
        "cache_prompt": os.environ.get("JEV_CACHE_PROMPT", "unset"),
        "mean_ms": statistics.fmean(samples),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "batch_mean_ms": statistics.fmean(batch_latencies),
        "throughput_req_s": len(samples) / (sum(batch_latencies) / 1000),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }, indent=2))


if __name__ == "__main__":
    main()
