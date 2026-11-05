import os
import sys
import types as pytypes
import unittest
from unittest.mock import patch

# The tests mock the MCP boundary and do not need the optional transport SDK.
fake_mcp = pytypes.ModuleType("mcp")
fake_mcp.ClientSession = object
fake_mcp.StdioServerParameters = object
fake_mcp_client = pytypes.ModuleType("mcp.client")
fake_mcp_stdio = pytypes.ModuleType("mcp.client.stdio")
fake_mcp_stdio.stdio_client = object
sys.modules.setdefault("mcp", fake_mcp)
sys.modules.setdefault("mcp.client", fake_mcp_client)
sys.modules.setdefault("mcp.client.stdio", fake_mcp_stdio)

from llm_service import generate_answer
from retrieval import load_knowledge_base, retrieve_entries


class FunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class FakeTypes:
    class Type:
        OBJECT, STRING, INTEGER = "OBJECT", "STRING", "INTEGER"

    class Schema:
        def __init__(self, **kwargs):
            pass

    class FunctionDeclaration:
        def __init__(self, **kwargs):
            pass

    class Tool:
        def __init__(self, **kwargs):
            pass

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            pass

    class Content:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Part:
        @staticmethod
        def from_text(**kwargs):
            return pytypes.SimpleNamespace(**kwargs)

        @staticmethod
        def from_function_response(**kwargs):
            return pytypes.SimpleNamespace(**kwargs)


class UnsupportedQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = load_knowledge_base()

    def _install_gemini(self, responses=None, failure=None):
        class Models:
            def __init__(self):
                self.responses = list(responses or [])

            def generate_content(self, **kwargs):
                if failure:
                    raise failure
                return self.responses.pop(0)

        fake_genai = pytypes.ModuleType("google.genai")
        fake_genai.Client = lambda **kwargs: pytypes.SimpleNamespace(models=Models())
        fake_google = pytypes.ModuleType("google")
        fake_google.__path__ = []
        fake_google.genai = fake_genai
        fake_types = pytypes.ModuleType("google.genai.types")
        fake_types.__dict__.update(FakeTypes.__dict__)
        return patch.dict(sys.modules, {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.types": fake_types,
        })

    @staticmethod
    def _tool_response(name, args):
        content = FakeTypes.Content(role="model", parts=[pytypes.SimpleNamespace(
            function_call=FunctionCall(name, args)
        )])
        return pytypes.SimpleNamespace(candidates=[pytypes.SimpleNamespace(content=content)], text="")

    @staticmethod
    def _text_response(text):
        content = FakeTypes.Content(role="model", parts=[])
        return pytypes.SimpleNamespace(candidates=[pytypes.SimpleNamespace(content=content)], text=text)

    def test_unrelated_sourdough_query_is_rejected(self):
        matches = retrieve_entries("How do I bake sourdough bread?", self.kb)
        self.assertEqual(matches, [])
        with self._install_gemini([self._tool_response("search_knowledge_base", {"query": "How do I bake sourdough bread?"})]), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), \
             patch("llm_service.call_tool", return_value={"matches": []}) as call:
            result = generate_answer("How do I bake sourdough bread?")
        self.assertIn("does not contain relevant information", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["tools_used"], [{"name": "search_knowledge_base", "status": "success"}])

    def test_supported_payment_error_is_relevant_and_grounded(self):
        matches = retrieve_entries("The payment API is returning HTTP 500 errors.", self.kb)
        self.assertEqual(matches[0]["id"], "KB001")
        sequence = [
            self._tool_response("search_knowledge_base", {"query": "The payment API is returning HTTP 500 errors."}),
            self._text_response("Review the documented payment API checks.")
        ]
        with self._install_gemini(sequence), patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), \
             patch("llm_service.call_tool", return_value={"matches": matches}) as call:
            result = generate_answer("The payment API is returning HTTP 500 errors.")
        self.assertEqual(result["source"], "gemini")
        self.assertEqual(result["sources"][0]["id"], "KB001")
        self.assertEqual(call.call_count, 1)

    def test_ambiguous_technical_query_has_no_diagnosis(self):
        self.assertEqual(retrieve_entries("API issue", self.kb), [])
        with self._install_gemini([self._tool_response("search_knowledge_base", {"query": "API issue"})]), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), \
             patch("llm_service.call_tool", return_value={"matches": []}):
            result = generate_answer("API issue")
        self.assertIn("No relevant knowledge-base information found", result["answer"])
        self.assertEqual(result["sources"], [])

    def test_gemini_api_failure_returns_safe_fallback(self):
        with self._install_gemini(failure=RuntimeError("API unavailable")), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            result = generate_answer("The payment API is returning HTTP 500 errors.")
        self.assertEqual(result["source"], "fallback")
        self.assertIn("can't provide a diagnosis or troubleshooting steps", result["answer"])
        self.assertIn("RuntimeError", result["error"])


if __name__ == "__main__":
    unittest.main()
