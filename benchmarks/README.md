# Benchmarks

The benchmark runner expects a local clone of the BBQ reference repository.
Pass its path with `--bbq-root`:

```bash
python benchmarks/run_bbq.py \\
  --bbq-root /path/to/jev-bbq-experiment \\
  --count 10000 \\
  --output benchmarks/results_bbq_10000.json
```

The latest benchmark is a deterministic 10,000-question sample (`random.seed(42)`).
The saved result includes row-level predictions, probabilities, confidence, and
latency. The runner does not download BBQ data automatically and does not upload
results anywhere.
