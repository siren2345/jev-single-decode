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
- NVIDIA GPU with CUDA support
- Sufficient VRAM for the selected model
- PyTorch and Transformers

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

Validation uses [`simonmesmith/jev-bbq-experiment`](https://github.com/simonmesmith/jev-bbq-experiment), with the frozen BBQ passage/question format. Results below use the same 100-question sample (`random.seed(42)`) and Qwen3-4B in FP16 on an RTX 5090.

| Prompt variant | Accuracy |
|---|---:|
| Current state plus extra output instruction | 80/100 (80%) |
| **Original BBQ state + Qwen chat template** | **86/100 (86%)** |
| Answer-format instruction moved to user message | 86/100 (86%) |
| Asking the model to emit `ans0`/`ans1`/`ans2` directly | 43/100 (43%) |

Canonical run metrics:

| Metric | Result |
|---|---:|
| Accuracy | **86/100 (86%)** |
| Ambiguous accuracy | 44/50 (88%) |
| Informative accuracy | 42/50 (84%) |
| Brier score | 0.2853 |
| ECE (10 bins) | 0.1462 |
| Mean confidence | 0.9932 |
| Mean confidence on incorrect answers | **~1.0000** |
| Mean latency | 36.6 ms |
| p50 / p95 latency | 33.0 / 38.4 ms |

The model is accurate enough to make the experiment interesting, but its raw
three-token softmax is visibly overconfident: incorrect predictions also tend
to have confidence near 1.0. The returned probability distribution should not
be treated as calibrated until a separate calibration split and evaluation are
added.

The reference implementation uses the original BBQ state without adding an extra system instruction. The 58,492-question dataset has not been run in full; this project intentionally stops at sampled validation while the implementation is experimental.

### 10,000-question follow-up

A larger seed-42 random sample was also run with the same canonical prompt and model:

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

The 10,000-question result is more representative than the initial 100-question result, but it is still a random validation sample rather than a full-dataset evaluation. The row-level output is stored in `benchmarks/results_bbq_10000.json`.

Warm GPU measurement after model loading, batch size 1, 12 sequential questions:

| Metric | Measurement |
|---|---:|
| Per question | 38 ms |
| Throughput | 26.6 questions/s |
| Execution mode | Sequential, no batching yet |

Latency depends on prompt length, GPU, dtype, and model. Batch inference is a planned throughput optimization; it should not be assumed to improve accuracy.

## Model substitution

The model is not hard-coded into the project design. Replace the model ID when constructing the judge:

```python
judge = QwenChoiceJudge("your-org/your-causal-model")
```

The replacement model must support a compatible chat template and direct next-token scoring for the selected labels.

## Repository layout

```text
src/jev_single_decode.py       Local Jev-compatible judge
benchmarks/            Prompt and benchmark scripts
```

## License

MIT. See [LICENSE](LICENSE).
