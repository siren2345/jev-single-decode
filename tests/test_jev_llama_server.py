from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from server import jev_llama_server as server


class ChoiceTests(unittest.TestCase):
    def test_probabilities_support_dynamic_labels(self) -> None:
        probabilities = server._probabilities(
            [
                {"token": " A", "logprob": -0.1},
                {"token": "B", "logprob": -1.1},
                {"token": "▁C", "logprob": -2.1},
                {"token": "ĠD", "logprob": -3.1},
            ],
            "ABCD",
        )

        self.assertEqual(len(probabilities), 4)
        self.assertTrue(math.isclose(sum(probabilities), 1.0))
        self.assertEqual(max(range(4), key=probabilities.__getitem__), 0)

    def test_probabilities_support_post_sampling_probs(self) -> None:
        probabilities = server._probabilities(
            [
                {"token": "C", "prob": 0.6},
                {"token": "A", "prob": 0.3},
                {"token": "B", "prob": 0.1},
            ],
            "ABC",
        )

        self.assertEqual(probabilities, [0.3, 0.1, 0.6])

    def test_question_prompt_uses_all_labels(self) -> None:
        criteria = {f"option_{index}": f"Value {index}" for index in range(26)}
        messages = server._question_messages("state", "Choose", criteria)

        self.assertIn("A. Value 0", messages[1]["content"])
        self.assertIn("Z. Value 25", messages[1]["content"])

    @patch.object(server, "_llama_choice")
    def test_predict_accepts_26_options(self, llama_choice) -> None:
        llama_choice.return_value = ([0.0] * 25 + [1.0], {})
        criteria = {f"option_{index}": f"Value {index}" for index in range(26)}

        result = server.predict(
            {
                "state": "state",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "Choose",
                        "criteria": criteria,
                    }
                },
            }
        )

        self.assertEqual(result["answers"]["q"]["choice"], "option_25")
        self.assertEqual(llama_choice.call_args.args[1], server.CHOICE_LABELS)

    def test_predict_rejects_fewer_than_2_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 to 26 criteria"):
            server.predict(
                {
                    "questions": {
                        "q": {
                            "type": "choice",
                            "instructions": "Choose",
                            "criteria": {"only": "Only"},
                        }
                    }
                }
            )

    def test_predict_rejects_more_than_26_options(self) -> None:
        criteria = {f"option_{index}": str(index) for index in range(27)}
        with self.assertRaisesRegex(ValueError, "2 to 26 criteria"):
            server.predict(
                {
                    "questions": {
                        "q": {
                            "type": "choice",
                            "instructions": "Choose",
                            "criteria": criteria,
                        }
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
