# jev-single-decode

An experiment asking whether Jev-like typed decisions can be implemented with an ordinary Transformer using prefill and exactly one decode step.
> This project is not affiliated with TypeSafe AI. Jev is a trademark of TypeSafe AI.

`jev-single-decode` runs a small causal language model locally, formats requests with the model's native chat template, and returns a typed `choice` answer with a probability distribution and confidence.

## Current status

Experimental. The current reference model is [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), but the implementation accepts a configurable Hugging Face model ID.

Implemented:

- Jev-compatible `state` + `questions` input shape
- `choice` questions with three criteria
- Qwen chat-template formatting (`system` / `user` separation)
- Qwen3 thinking disabled for direct one-token choice selection
- Prefill plus one next-token decision over `A`, `B`, and `C`
- `probabilities` and `confidence` in the Choice response

Currently not implemented:

- `score`
- `noul`
- Free-form explanations
- Model weights bundled in this repository

## Quickstart

### Requirements

- Python 3.11+
- PyTorch and Transformers
- A local causal language model supported by the selected runtime

The reference Transformers implementation currently places the model on
CUDA, so an NVIDIA GPU with sufficient VRAM is required for that path. The
llama.cpp API path does not require NVIDIA hardware; it can use any backend
supported by the installed llama.cpp build, including CPU-only execution.

Install dependencies:

```bash
pip install torch transformers safetensors huggingface_hub
```

Load the model from Hugging Face and run a Jev-shaped request:

```python
from jev_single_decode import LocalChoiceJudge

judge = LocalChoiceJudge("Qwen/Qwen3-4B")

request = {
    "model": "jev-1.13.0",
    "state": (
        "Answer each question using only its accompanying passage. "
        "If the passage does not determine the answer, choose the corresponding uncertainty option."
    ),
    "questions": {
        "who": {
            "type": "choice",
            "instructions": "Passage: Alice went home.\n\nQuestion: Who went home?",
            "criteria": {
                "ans0": "Alice",
                "ans1": "Bob",
                "ans2": "Can't be determined"
            }
        }
    }
}

response = judge.predict(request)
print(response)
```

Example response:

```json
{
  "model": "Qwen/Qwen3-4B",
  "answers": {
    "who": {
      "type": "choice",
      "choice": "ans0",
      "probabilities": {
        "ans0": 0.98,
        "ans1": 0.01,
        "ans2": 0.01
      },
      "confidence": 0.98
    }
  }
}
```

`confidence` is currently the maximum probability among the supplied criteria. It is a local softmax-derived value, not a claim that it is calibrated like the hosted Jev model.

## Jev compatibility

The input follows the public Jev shape:

```json
{
  "model": "jev-1.13.0",
  "state": "shared state",
  "questions": {
    "question_id": {
      "type": "choice",
      "instructions": "question text",
      "criteria": {
        "label_a": "meaning of option A",
        "label_b": "meaning of option B",
        "label_c": "meaning of option C"
      }
    }
  }
}
```

The output preserves the criteria keys. Any three-key criteria mapping can be used; the implementation does not require the names `ans0`, `ans1`, and `ans2`.

## Prompt format

For Qwen3, the request is rendered through `apply_chat_template`:

```text
<|im_start|>system
[state]<|im_end|>
<|im_start|>user
Passage: ...

Question: ...

Options:
A. ...
B. ...
C. ...

Answer:<|im_end|>
<|im_start|>assistant
<think>

</think>
```

The empty thinking block is produced by Qwen3 when `enable_thinking=False`. The next-token logits for raw `A`, `B`, and `C` are compared; no free-form text is generated.

## BBQ benchmark

Validation uses [`simonmesmith/jev-bbq-experiment`](https://github.com/simonmesmith/jev-bbq-experiment), with the frozen BBQ passage/question format. The latest benchmark uses Qwen3-4B in FP16 on an RTX 5090, the original BBQ state, Qwen chat template, thinking disabled, and one A/B/C decision.

### Latest benchmark: 10,000 random questions

The sample uses `random.seed(42)` and is not the full 58,492-question dataset.

| Metric | Result |
|---|---:|
| Accuracy | **9,053/10,000 (90.53%)** |
| Ambiguous accuracy | 4,679/5,018 (93.24%) |
| Informative accuracy | 4,374/4,982 (87.80%) |
| Brier score | 0.1828 |
| ECE (10 bins) | 0.0890 |
| Mean confidence | 0.9943 |
| Mean latency | 33.5 ms |
| p50 / p95 latency | 32.3 / 42.3 ms |

The raw three-token softmax is still overconfident and should not be treated as calibrated until a separate calibration split and evaluation are added. The row-level output is stored in `benchmarks/results_bbq_10000.json`.

Latency depends on prompt length, GPU, dtype, and model. Batch inference is a planned throughput optimization and should not be assumed to improve accuracy.

Run the benchmark with a local BBQ clone:

```bash
python benchmarks/run_bbq.py \\
  --bbq-root /path/to/jev-bbq-experiment \\
  --count 10000 \\
  --output benchmarks/results_bbq_10000.json
```

## llama.cpp API server

For a dedicated single-decode llama.cpp process, start the official
`llama-server` binary directly with the relevant options:

```bash
llama-server \\
  --model /path/to/model.gguf \\
  --alias jev-single-decode \\
  --host 127.0.0.1 --port 8080 \\
  --ctx-size 8192 --parallel 1 \\
  --gpu-layers 999 --flash-attn on --jinja \\
  --reasoning off --spec-type none \\
  --batch-size 2048 --ubatch-size 512 \\
  --cache-type-k q8_0 --cache-type-v q8_0 \\
  --temperature 0 --top-k 0 --top-p 1 --min-p 0 \\
  --predict 1 --metrics --slots --no-ui
```

The exact binary path can be used instead of `llama-server`. Increase
`--parallel` only when concurrent API requests are needed. `--batch-size` and
`--ubatch-size` affect prompt prefill, which is the costly part even though the
answer decode is only one token.

After llama.cpp is running, start `server/jev_llama_server.py`. It is a small
HTTP adapter that calls llama.cpp's OpenAI-compatible
`/v1/chat/completions` endpoint with `max_tokens=1` and reads `top_logprobs` for
the A/B/C decision:

```bash
LLAMA_CPP_URL=http://127.0.0.1:8080 \\
LLAMA_CPP_MODEL=jev-single-decode \\
python server/jev_llama_server.py
```

The adapter exposes the Jev-shaped request and response at:

- `POST /v1/systemone`
- `POST /v1/jev` (alias)
- `POST /v1/choices` (alias)

The adapter listens on `127.0.0.1:8090` by default. Set `JEV_HOST`, `JEV_PORT`,
`JEV_TOP_LOGPROBS`, or `LLAMA_TIMEOUT` to change the defaults. It does not
load model weights itself; llama.cpp remains the inference server.

The configured llama.cpp model must be able to produce A/B/C as candidates in
the first output token. Reasoning-oriented models that spend the first token
on hidden reasoning are not suitable for this one-token endpoint.

## Model substitution

The model is not hard-coded into the project design. Replace the model ID when constructing the judge:

```python
judge = LocalChoiceJudge("your-org/your-causal-model")
```

The replacement model must support a compatible chat template and direct next-token scoring for the selected labels.

## Repository layout

```text
src/jev_single_decode.py       Local Jev-compatible judge
server/jev_llama_server.py     llama.cpp-backed Jev-compatible HTTP API
benchmarks/                     Latest BBQ benchmark scripts and result
```

## License

MIT. See [LICENSE](LICENSE).
