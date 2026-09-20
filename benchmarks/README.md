# Benchmarks

The prompt comparison script expects a local clone of the BBQ reference repository.
Set `BBQ_ROOT` if it is not in the current working directory:

```bash
BBQ_ROOT=/path/to/jev-bbq-experiment \
python benchmarks/compare_prompts_qwen3_4b.py
```

The script evaluates four prompt variants on a deterministic 100-question sample
(`random.seed(42)`). The canonical run is saved as
`results_bbq_100.json` and includes row-level predictions, probabilities,
confidence, and latency. It does not download the BBQ data automatically and it does
not upload results anywhere.
