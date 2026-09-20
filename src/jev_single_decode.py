"""Local Jev-compatible choice-only judge using Qwen3-4B.

Inference is prefill plus one greedy decode decision. The model is prompted
with Qwen's chat template, thinking disabled, and only choice is returned.
"""
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"
STATE = (
    "Answer each question using only its accompanying passage. "
    "If the passage does not determine the answer, choose the corresponding "
    "uncertainty option."
)

class LocalChoiceJudge:
    def __init__(self, model_id: str = MODEL_ID, dtype: torch.dtype = torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to("cuda")
        self.model.eval()
        self.letter_ids = [
            self.tokenizer.encode(letter, add_special_tokens=False)[0]
            for letter in "ABC"
        ]

    def _prompt(self, row: dict[str, Any], state: str = STATE) -> str:
        options = row["options"]
        messages = [
            {"role": "system", "content": state},
            {"role": "user", "content": (
                f"Passage: {row['context']}\n\n"
                f"Question: {row['question']}\n\n"
                f"Options:\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\n\n"
                "Answer:"
            )},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @torch.inference_mode()
    def choice_distribution(
        self, row: dict[str, Any], state: str = STATE
    ) -> tuple[int, list[float], float]:
        prompt = self._prompt(row, state=state)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        logits = self.model(input_ids=inputs["input_ids"].to("cuda")).logits[0, -1]
        scores = logits[self.letter_ids]
        probabilities = torch.softmax(scores.float(), dim=0)
        choice_index = int(torch.argmax(probabilities).item())
        values = [float(value) for value in probabilities.cpu()]
        return choice_index, values, max(values)

    def choice(self, row: dict[str, Any], state: str = STATE) -> int:
        return self.choice_distribution(row, state=state)[0]

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload.get("questions"), dict):
            raise ValueError("Jev payload requires a questions object")
        state = payload.get("state", STATE)
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        answers = {}
        for question_id, question in payload["questions"].items():
            if question.get("type") != "choice":
                raise ValueError(
                    f"{question_id}: only type=choice is supported; "
                    "score/noul are currently unsupported"
                )
            instructions = question.get("instructions", question.get("question"))
            criteria = question.get("criteria")
            if not isinstance(instructions, str) or not isinstance(criteria, dict):
                raise ValueError(f"{question_id}: instructions and criteria are required")
            marker = "\n\nQuestion: "
            if marker not in instructions:
                raise ValueError(f"{question_id}: expected 'Passage: ... Question: ...' instructions")
            context, prompt_question = instructions.split(marker, 1)
            context = context.removeprefix("Passage: ")
            criterion_keys = list(criteria)
            if len(criterion_keys) != 3:
                raise ValueError(f"{question_id}: exactly three criteria are required")
            row = {
                "context": context,
                "question": prompt_question,
                "options": [criteria[key] for key in criterion_keys],
            }
            choice_index, probabilities, confidence = self.choice_distribution(row, state=state)
            answers[question_id] = {
                "type": "choice",
                "choice": criterion_keys[choice_index],
                "probabilities": {
                    key: probabilities[index]
                    for index, key in enumerate(criterion_keys)
                },
                "confidence": confidence,
            }
        return {"model": MODEL_ID, "answers": answers}


def load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
