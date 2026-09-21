"""Small Jev-compatible HTTP server backed by llama.cpp server.

The upstream llama.cpp server is expected at LLAMA_CPP_URL (default:
http://127.0.0.1:8080). This adapter uses /v1/chat/completions with exactly
a single output token and reads the returned probabilities for A-Z.
"""
from __future__ import annotations

import json
import os
import string
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
CHOICE_LABELS = string.ascii_uppercase


def _error(message: str, status: int = 400) -> tuple[int, dict[str, Any]]:
    return status, {"error": {"message": message, "type": "invalid_request_error"}}


def _token_letter(token: str, labels: str) -> str | None:
    """Return an allowed label for a token containing one answer letter."""
    cleaned = token.strip().replace("▁", " ").replace("Ġ", " ").strip()
    return cleaned if cleaned in labels else None


def _probabilities(logprobs: list[dict[str, Any]], labels: str) -> list[float]:
    """Extract and renormalize allowed-label probabilities."""
    import math

    scores: dict[str, float] = {}
    probabilities: dict[str, float] = {}
    for item in logprobs:
        letter = _token_letter(str(item.get("token", "")), labels)
        if letter is not None:
            if "prob" in item:
                probabilities[letter] = max(
                    probabilities.get(letter, 0.0), float(item["prob"])
                )
            elif "logprob" in item:
                scores[letter] = max(
                    scores.get(letter, float("-inf")), float(item["logprob"])
                )
    available = probabilities if probabilities else scores
    missing = [letter for letter in labels if letter not in available]
    if missing:
        raise RuntimeError(
            "llama.cpp did not return all choice labels; "
            f"missing={','.join(missing)}. Use a recent llama.cpp build and "
            "increase JEV_TOP_LOGPROBS if necessary."
        )
    if probabilities:
        values = [probabilities[letter] for letter in labels]
        total = sum(values)
        return [value / total for value in values]
    maximum = max(scores.values())
    values = [math.exp(scores[letter] - maximum) for letter in labels]
    total = sum(values)
    return [value / total for value in values]


def _question_messages(
    state: str, instructions: str, criteria: dict[str, Any]
) -> list[dict[str, str]]:
    options = list(criteria.values())
    option_lines = "\n".join(
        f"{CHOICE_LABELS[index]}. {option}" for index, option in enumerate(options)
    )
    user = (
        f"{instructions}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Answer:"
    )
    return [{"role": "system", "content": state}, {"role": "user", "content": user}]


def _llama_choice(
    messages: list[dict[str, str]], labels: str
) -> tuple[list[float], dict[str, Any]]:
    top_logprobs = max(int(os.environ.get("JEV_TOP_LOGPROBS", "50")), len(labels))
    grammar = "root ::= " + " | ".join(json.dumps(label) for label in labels)
    logit_bias = [[label, 100.0] for label in labels]
    payload = {
        "model": MODEL_NAME or "local",
        "messages": messages,
        "max_tokens": 1,
        "temperature": 1,
        "top_k": 0,
        "top_p": 1,
        "min_p": 0,
        "repeat_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "post_sampling_probs": True,
        "min_keep": len(labels),
        "logit_bias": logit_bias,
        "grammar": grammar,
        "seed": 0,
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
        candidates = content.get("top_probs", content.get("top_logprobs"))
        probabilities = _probabilities(candidates, labels)
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
            raise ValueError(
                f"{question_id}: only type=choice is supported; "
                "score/noul are currently unsupported"
            )
        instructions = question.get("instructions", question.get("question"))
        criteria = question.get("criteria")
        if (
            not isinstance(instructions, str)
            or not isinstance(criteria, dict)
            or not 2 <= len(criteria) <= len(CHOICE_LABELS)
        ):
            raise ValueError(
                f"{question_id}: instructions and 2 to 26 criteria are required"
            )
        keys = list(criteria)
        labels = CHOICE_LABELS[: len(criteria)]
        probabilities, _raw = _llama_choice(
            _question_messages(state, instructions, criteria), labels
        )
        index = max(range(len(criteria)), key=probabilities.__getitem__)
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
