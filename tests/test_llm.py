"""Smoke tests for the LLM interface module."""

import unittest
from app.llm import LocalLLM, generate_response, DEFAULT_MODEL_NAME


class TestLLMInterface(unittest.TestCase):
    """Smoke test suite for LLM module exports and interface."""

    def test_imports_and_defaults(self):
        """Verify module components and defaults are properly exported and configurable."""
        self.assertEqual(DEFAULT_MODEL_NAME, "qwen3-4b")
        llm = LocalLLM()
        self.assertEqual(llm.model_name, "qwen3-4b")
        self.assertIsNotNone(llm)

    def test_llm_callable(self):
        """Verify the LLM interface can be called and produces a valid response string."""
        response = generate_response("Say hello in three words.", max_tokens=50)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response.strip()) > 0)


if __name__ == "__main__":
    unittest.main()
