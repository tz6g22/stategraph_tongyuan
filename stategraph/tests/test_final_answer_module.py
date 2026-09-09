import unittest

from stategraph.final_answer import FINAL_SYSTEM_PROMPT, build_answer_input, parse_answer


class FinalAnswerModuleTests(unittest.TestCase):
    def test_prompt_uses_only_frozen_context_and_marks_stale_policy(self):
        row = {
            "query": "Can it proceed?",
            "query_type": "action",
            "response_policy": "reject_stale_premise",
            "final_context": ["Stale premise rejected", "STATE: CURRENT"],
        }
        messages = build_answer_input(row, improved=True)
        self.assertEqual(messages[0]["content"], FINAL_SYSTEM_PROMPT)
        self.assertIn("reject_stale_premise", messages[1]["content"])
        self.assertIn("STATE: CURRENT", messages[1]["content"])
        self.assertNotIn("gold", messages[1]["content"].casefold())

    def test_answer_parser_removes_markdown_wrapper_only(self):
        self.assertEqual(parse_answer("```text\nNo.\n```"), "No.")
        self.assertEqual(parse_answer("  No.  "), "No.")
