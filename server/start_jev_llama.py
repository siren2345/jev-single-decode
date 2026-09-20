"""Start llama.cpp with the Jev single-decode preset on any platform.

Usage:
    python server/start_jev_llama.py --model /path/to/model.gguf

The llama-server executable must be available on PATH, or pass --executable.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--executable", default="llama-server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--threads-batch", type=int, default=16)
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"model was not found: {args.model}")
    executable = shutil.which(args.executable)
    if executable is None:
        parser.error(f"llama-server was not found on PATH: {args.executable}")

    command = [
        executable,
        "--model", str(args.model),
        "--alias", "jev-single-decode",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--ctx-size", str(args.context * args.parallel),
        "--parallel", str(args.parallel),
        "--gpu-layers", "999",
        "--flash-attn", "on",
        "--jinja",
        "--reasoning", "off",
        "--spec-type", "none",
        "--batch-size", str(args.batch),
        "--ubatch-size", str(args.ubatch),
        "--threads", str(args.threads),
        "--threads-batch", str(args.threads_batch),
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--temperature", "0",
        "--top-k", "0",
        "--top-p", "1",
        "--min-p", "0",
        "--predict", "1",
        "--metrics",
        "--slots",
        "--no-ui",
    ]
    print("llama.cpp single-decode preset", flush=True)
    print("command:", " ".join(command), flush=True)
    return subprocess.call(command, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
