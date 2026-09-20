"""Small Jev-compatible HTTP server backed by llama.cpp server.

The upstream llama.cpp server is expected at LLAMA_CPP_URL (default:
http://127.0.0.1:8080). This adapter uses /v1/chat/completions with exactly
one output token and reads the returned top-logprobs for A/B/C.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LLAMA_CPP_URL = os.environ.get("LLAMA_CPP_URL", "http://127.0.0.1:8080").rstrip("/")
MODEL_NAME = os.environ.get("LLAMA_CPP_MODEL", "")
HOST = os.environ.get("JEV_HOST", "127.0.0.1")
PORT = int(os.environ.get("JEV_PORT", "8090"))
DEFAULT_STATE = (
    "Answer each question using only its accompanying passage. "
    "If the passage does not determine the answer, choose the corresponding "
    "uncertainty option."
)


def _error(message: str, status: int = 400) -> tuple[int, dict[str, Any]]:
    return status, {"error": {"message": message, "type": "invalid_request_error"}}


def _token_letter(token: str) -> str | None:
    """Return A/B/C only for a token consisting of one answer letter."""
    cleaned = token.strip().replace("▁", " ").replace("Ġ", " ").strip()
    return cleaned if cleaned in {"A", "B", "C"} else None


def _probabilities(logprobs: list[dict[str, Any]]) -> list[float]:
    """Extract and renormalize A/B/C probabilities from top-logprobs."""
    import math

    scores: dict[str, float] = {}
    for item in logprobs:
        letter = _token_letter(str(item.get("token", "")))
        if letter is not None:
            scores[letter] = max(scores.get(letter, float("-inf")), float(item["logprob"]))
    missing = [letter for letter in "ABC" if letter not in scores]
    if missing:
        raise RuntimeError(
            "llama.cpp did not return all A/B/C candidates in top_logprobs; "
            f"missing={','.join(missing)}. Increase JEV_TOP_LOGPROBS."
        )
    maximum = max(scores.values())
    values = [math.exp(scores[letter] - maximum) for letter in "ABC"]
    total = sum(values)
    return [value / total for value in values]


def _question_messages(state: str, instructions: str, criteria: dict[str, Any]) -> list[dict[str, str]]:
    options = list(criteria.values())
    user = (
        f"{instructions}\n\n"
        f"Options:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\n\n"
        "Answer:"
    )
    return [{"role": "system", "content": state}, {"role": "user", "content": user}]


def _llama_choice(messages: list[dict[str, str]]) -> tuple[list[float], dict[str, Any]]:
    payload = {
        "model": MODEL_NAME or "local",
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": int(os.environ.get("JEV_TOP_LOGPROBS", "50")),
        "stream": False,
    }
    request = Request(
        f"{LLAMA_CPP_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(os.environ.get("LLAMA_TIMEOUT", "120"))) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"llama.cpp request failed: {exc}") from exc
    try:
        content = result["choices"][0]["logprobs"]["content"][0]
        probabilities = _probabilities(content["top_logprobs"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("unexpected llama.cpp chat completion logprobs response") from exc
    return probabilities, result


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    questions = payload.get("questions")
    if not isinstance(questions, dict):
        raise ValueError("Jev payload requires a questions object")
    state = payload.get("state", DEFAULT_STATE)
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    answers: dict[str, Any] = {}
    for question_id, question in questions.items():
        if not isinstance(question, dict) or question.get("type") != "choice":
            raise ValueError(f"{question_id}: only type=choice is supported; score/noul are currently unsupported")
        instructions = question.get("instructions", question.get("question"))
        criteria = question.get("criteria")
        if not isinstance(instructions, str) or not isinstance(criteria, dict) or len(criteria) != 3:
            raise ValueError(f"{question_id}: instructions and exactly three criteria are required")
        keys = list(criteria)
        probabilities, _raw = _llama_choice(_question_messages(state, instructions, criteria))
        index = max(range(3), key=probabilities.__getitem__)
        answers[question_id] = {
            "type": "choice",
            "choice": keys[index],
            "probabilities": {key: probabilities[i] for i, key in enumerate(keys)},
            "confidence": probabilities[index],
        }
    return {"model": MODEL_NAME or "llama.cpp", "answers": answers}


class Handler(BaseHTTPRequestHandler):
    server_version = "jev-single-decode/llama.cpp"

    def _send(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/health", "/v1/health"}:
            self._send(200, {"status": "ok", "backend": "llama.cpp", "llama_cpp_url": LLAMA_CPP_URL})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/v1/systemone", "/v1/jev", "/v1/choices"}:
            self._send(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            self._send(200, predict(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            self._send(*_error(str(exc), 400))
        except RuntimeError as exc:
            self._send(*_error(str(exc), 502))


def main() -> None:
    print(f"jev llama.cpp API listening on http://{HOST}:{PORT}", flush=True)
    print(f"backend: {LLAMA_CPP_URL}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
