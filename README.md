# jev-single-decode

A small llama.cpp-backed HTTP adapter that exposes a Jev-compatible `choice`
API. The adapter tests whether a typed choice decision can be served by an
ordinary causal model using prompt prefill and exactly one output token.

> This project is not affiliated with TypeSafe AI. Jev is a trademark of
> TypeSafe AI.

## What this repository contains

- `server/jev_llama_server.py`: Jev-shaped HTTP API backed by llama.cpp
- `POST /v1/systemone`: primary endpoint
- `POST /v1/jev` and `POST /v1/choices`: aliases
- `GET /health`: adapter health check

The adapter sends one request per question to llama.cpp's OpenAI-compatible
`/v1/chat/completions` endpoint with `max_tokens=1`. It reads `top_logprobs`
for the A/B/C answer tokens, renormalizes those three values, and returns
`choice`, `probabilities`, and `confidence` while preserving the input
criteria keys.

This repository does not bundle model weights. Any llama.cpp-compatible model
can be used, provided it can produce A/B/C as candidates in its first output
token. Reasoning-oriented models that spend the first token on hidden
reasoning are not suitable for this endpoint.

## Requirements

- Python 3.11+
- A llama.cpp `llama-server` build
- A GGUF model supported by that build
- Any compute backend supported by llama.cpp, including CPU-only execution

## Start the servers

First start llama.cpp with one-token deterministic decoding. The options below
are a reference configuration; adjust GPU and context settings for the model
and hardware:

```bash
llama-server \\
  --model /path/to/model.gguf \\
  --alias jev-single-decode \\
  --host 127.0.0.1 --port 8080 \\
  --ctx-size 40960 --parallel 20 \\
  --gpu-layers 999 --flash-attn on --jinja \\
  --reasoning off --spec-type none \\
  --batch-size 4096 --ubatch-size 1024 \\
  --cache-type-k q8_0 --cache-type-v q8_0 \\
  --temperature 0 --top-k 0 --top-p 1 --min-p 0 \\
  --predict 1 --metrics --slots --no-ui
```

Then start the adapter in a second terminal:

```bash
LLAMA_CPP_URL=http://127.0.0.1:8080 \\
LLAMA_CPP_MODEL=jev-single-decode \\
python server/jev_llama_server.py
```

The adapter listens on `127.0.0.1:8090` by default. Configure it with
`LLAMA_CPP_URL`, `LLAMA_CPP_MODEL`, `JEV_HOST`, `JEV_PORT`,
`JEV_TOP_LOGPROBS`, and `LLAMA_TIMEOUT`.

## Request and response

```json
{
  "model": "jev-1.13.0",
  "state": "Answer using only the passage.",
  "questions": {
    "who": {
      "type": "choice",
      "instructions": "Passage: Alice went home.\n\nQuestion: Who went home?",
      "criteria": {
        "ans0": "Alice",
        "ans1": "Bob",
        "ans2": "Cannot be determined"
      }
    }
  }
}
```

The response contains the same criteria keys:

```json
{
  "model": "jev-single-decode",
  "answers": {
    "who": {
      "type": "choice",
      "choice": "ans0",
      "probabilities": {"ans0": 0.98, "ans1": 0.01, "ans2": 0.01},
      "confidence": 0.98
    }
  }
}
```

`score` and `noul` are currently unsupported. The returned probabilities are
softmax-derived and API-compatible, but they should not be treated as
calibrated probabilities without a separate calibration evaluation.

## Architecture

```text
client
  -> jev_llama_server.py :8090
  -> llama-server :8080
  -> GGUF model
```

The adapter is intentionally small and uses only the Python standard library.
llama.cpp remains responsible for model loading, prompt formatting, inference,
and hardware acceleration.

## Long-prompt BBQ benchmark

The llama.cpp path was also run against the same deterministic 10,000-question
sample, using Qwen3-4B-Q4_K_M on an RTX 5090. This is an end-to-end HTTP result:

| Metric | llama.cpp result |
|---|---:|
| Accuracy | **8,840/10,000 (88.40%)** |
| Ambiguous accuracy | 92.17% |
| Informative accuracy | 84.60% |
| Brier score | 0.2191 |
| ECE (10 bins) | 0.1067 |
| Mean request latency | 628 ms |
| p50 / p95 latency | 537 / 1,063 ms |
| Throughput | 31.83 req/s |

This run used `parallel 100`, 2,048 tokens per slot, q8 KV cache,
`cache_prompt=off`, `top_logprobs=50`, and client concurrency 20. The raw
result is stored in `benchmarks/results_bbq_10000_llama_p100.json`. Client
concurrency is part of the measurement: it reports concurrent throughput and
must be compared using the same client concurrency and prompt distribution.

## llama.cpp API server

`server/jev_llama_server.py` provides a small HTTP adapter for a running
llama.cpp server. It uses llama.cpp's OpenAI-compatible
`/v1/chat/completions` endpoint with `max_tokens=1` and reads the returned
`top_logprobs` for the A/B/C decision. The adapter exposes the same Jev-shaped
request and response at `POST /v1/systemone` (with `/v1/jev` and
`/v1/choices` as aliases).

Start it after starting llama.cpp:

```bash
LLAMA_CPP_URL=http://127.0.0.1:8080 \\
LLAMA_CPP_MODEL=your-model \\
python server/jev_llama_server.py
```

For a dedicated single-decode llama.cpp process, use
`server/start-jev-llama.ps1`. Its preset enables Flash Attention, disables
reasoning and speculative decoding, uses deterministic sampling, and keeps
20 server slots by default:

```powershell
pwsh -File server/start-jev-llama.ps1 `
  -Model C:\path\to\model.gguf
```

Use `-DryRun` first to print the resolved command. The defaults are tuned for
one-token decisions and allow up to 20 concurrent requests with 2048 tokens
per slot. `-Context` is per slot, so increasing it also increases total KV
cache as `Context * Parallel`. Reduce `-Parallel` on smaller GPUs. `-Batch`
and `-UBatch` affect prompt prefill, which is the costly part even though the
answer decode is only one token. The preset defaults to `f16/f16` KV cache for
logit precision; use `-CacheTypeK q8_0 -CacheTypeV q8_0` only when lower VRAM
use or higher throughput is more important.

For long BBQ-style prompts on a 32 GB GPU, the measured high-throughput profile
was:

```powershell
pwsh -File server/start-jev-llama.ps1 `
  -Model C:\path\to\Qwen3-4B-Q4_K_M.gguf `
  -Context 2048 -Parallel 100 `
  -Batch 4096 -UBatch 1024 `
  -CacheTypeK q8_0 -CacheTypeV q8_0
```

In a 100-question BBQ tuning sample, this profile reached 46.4 req/s at client
concurrency 20, compared with 34.3 req/s for `batch 2048 / ubatch 512`.
At concurrency 50 and 100, throughput saturated around 44.5 req/s while
latency increased, so `parallel 100` is a capacity setting rather than a
single-request latency optimization.

To measure the llama.cpp path with the same end-to-end API boundary:

```powershell
JEV_CACHE_PROMPT=1 python benchmarks/bench_llama_api.py \
  --warmup 10 --count 100 --concurrency 20
```

Set `JEV_CACHE_PROMPT=0` for the control run. The benchmark reports request
latency, batch latency, throughput, and the cache mode. Compare those numbers
only on the same machine, model, prompt set, concurrency, and warmup policy.

The adapter listens on `127.0.0.1:8090` by default. Set `JEV_HOST`, `JEV_PORT`,
`JEV_TOP_LOGPROBS`, or `LLAMA_TIMEOUT` to change the defaults. It does not
load model weights itself; llama.cpp remains the inference server.

The configured llama.cpp model must be able to produce A/B/C as candidates in
the first output token. Reasoning-oriented models that spend the first token
on hidden reasoning are not suitable for this one-token endpoint.

For the BBQ HTTP benchmark, use `benchmarks/run_bbq_api.py` with a local clone
of `simonmesmith/jev-bbq-experiment`:

```powershell
python benchmarks/run_bbq_api.py `
  --bbq-root ..\jev-bbq-experiment `
  --count 10000 --concurrency 20 `
  --output benchmarks/results_bbq_10000_llama_p100.json
```

## Reproducible llama.cpp target

The llama.cpp reference model in this repository is **Qwen3-4B-Q4_K_M** from
[`Qwen/Qwen3-4B-GGUF`](https://huggingface.co/Qwen/Qwen3-4B-GGUF). Do not use
27B models for the headline comparison. Report the model quantization, GPU,
llama.cpp version, warmup count, concurrency, and sample count with every
benchmark result.

## Repository layout

```text
server/jev_llama_server.py     llama.cpp-backed Jev-compatible HTTP API
server/start-jev-llama.ps1     tuned llama.cpp launcher
benchmarks/                     Latest BBQ benchmark scripts and result
```

## License

MIT. See [LICENSE](LICENSE).
