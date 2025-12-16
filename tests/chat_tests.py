"""
Chat completion tests for LLM interfaces.
Tests basic chat functionality, structured outputs, and error handling.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

load_dotenv()

from llm_utils.wrapper import LLMWrapper
from llm_utils.openai_interface import OpenAIInterface
from llm_utils.claude_interface import ClaudeInterface
from llm_utils.gemini_interface import GeminiInterface


# Test fixtures
@pytest.fixture
def llm():
    return LLMWrapper()


@pytest.fixture
def system_message():
    return "You are a helpful assistant."


@pytest.fixture
def simple_message():
    return ["What is 2+2? Answer with just the number."]


@pytest.fixture
def simple_format():
    return [{
        "name": "math_response",
        "type": "json_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"],
            "additionalProperties": False
        }
    }]


@pytest.fixture
def structured_response_format():
    """Response format from demo.ipynb"""
    return [
        {
            "name": "example_response1",
            "type": "json_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "FIVE_YEAR_OLD_EXPLANATION": {"type": "string"},
                    "COLLEGE_STUDENT_EXPLANATION": {"type": "string"}
                },
                "required": ["FIVE_YEAR_OLD_EXPLANATION", "COLLEGE_STUDENT_EXPLANATION"],
                "additionalProperties": False
            }
        }
    ]


# ============== OpenAI Chat Tests ==============

class TestOpenAIChat:
    """Tests for OpenAI chat completions"""

    def test_openai_basic_chat(self, llm, system_message, simple_message, simple_format):
        """Test basic OpenAI chat completion with structured output"""
        response = llm.ask(
            system_message=system_message,
            ids=["0"],
            user_messages=simple_message,
            response_formats=simple_format,
            model="gpt-4o-mini",
            n_workers=1,
        )

        assert "0" in response
        assert "answer" in response["0"]
        assert "4" in response["0"]["answer"]

    def test_openai_structured_output(self, llm, system_message, structured_response_format):
        """Test OpenAI with complex structured output"""
        response = llm.ask(
            system_message=system_message,
            ids=["0"],
            user_messages=["Explain gravity in one sentence for each audience."],
            response_formats=structured_response_format,
            model="gpt-4o-mini",
            n_workers=1,
        )

        assert "0" in response
        assert "FIVE_YEAR_OLD_EXPLANATION" in response["0"]
        assert "COLLEGE_STUDENT_EXPLANATION" in response["0"]
        assert len(response["0"]["FIVE_YEAR_OLD_EXPLANATION"]) > 0
        assert len(response["0"]["COLLEGE_STUDENT_EXPLANATION"]) > 0


class TestOpenAIErrorHandling:
    """Tests for OpenAI error handling"""

    def test_openai_auth_error(self):
        """Test that authentication errors are handled properly"""
        with patch.dict('os.environ', {'OPENAI_KEY': 'invalid-key'}):
            interface = OpenAIInterface()
            response = interface._chat_completion_call(
                model="gpt-4o-mini",
                message="Hello",
                web_search=False,
                max_tokens=100,
                max_tool_calls=3,
                tool_choice=None,
                system_message="You are helpful.",
                response_format=None,
                verbosity="medium",
                reasoning=None
            )

            assert isinstance(response, dict)
            assert "error" in response
            assert response.get("error_type") in ["auth", "bad_request", "max_retries"]

    def test_openai_bad_model_error(self, llm, system_message, simple_message, simple_format):
        """Test that invalid model errors are handled"""
        response = llm.ask(
            system_message=system_message,
            ids=["0"],
            user_messages=simple_message,
            response_formats=simple_format,
            model="gpt-nonexistent-model",
            n_workers=1,
        )

        assert "0" in response
        assert "error" in response["0"]


# ============== Claude Chat Tests ==============

class TestClaudeChat:
    """Tests for Claude chat completions"""

    def test_claude_basic_chat(self, llm, system_message, simple_message, simple_format):
        """Test basic Claude chat completion with structured output"""
        response = llm.ask(
            system_message=system_message,
            ids=["0"],
            user_messages=simple_message,
            response_formats=simple_format,
            model="claude-3-5-haiku-latest",
            n_workers=1,
        )

        assert "0" in response
        assert "answer" in response["0"]
        assert "4" in response["0"]["answer"]

    def test_claude_structured_output(self, llm, system_message, structured_response_format):
        """Test Claude with complex structured output"""
        response = llm.ask(
            system_message=system_message,
            ids=["0"],
            user_messages=["Explain gravity in one sentence for each audience."],
            response_formats=structured_response_format,
            model="claude-3-5-haiku-latest",
            n_workers=1,
        )

        assert "0" in response
        assert "FIVE_YEAR_OLD_EXPLANATION" in response["0"]
        assert "COLLEGE_STUDENT_EXPLANATION" in response["0"]


class TestClaudeErrorHandling:
    """Tests for Claude error handling"""

    def test_claude_auth_error(self):
        """Test that authentication errors are handled properly"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'invalid-key'}):
            interface = ClaudeInterface()
            response = interface._chat_completion_call(
                model="claude-3-5-haiku-latest",
                message="Hello",
                web_search=False,
                max_tokens=100,
                max_tool_calls=3,
                tool_choice=None,
                system_message="You are helpful.",
                response_format=None,
                verbosity="medium",
                reasoning=None
            )

            assert isinstance(response, dict)
            assert "error" in response
            assert response.get("error_type") in ["auth", "bad_request", "max_retries"]

    def test_claude_bad_model_error(self, llm, system_message, simple_message, simple_format):
        """Test that invalid model errors are handled"""
        response = llm.ask(
            system_message=system_message,
            ids=["0"],
            user_messages=simple_message,
            response_formats=simple_format,
            model="claude-nonexistent-model",
            n_workers=1,
        )

        assert "0" in response
        assert "error" in response["0"]


# ============== Gemini Chat Tests ==============

class TestGeminiChat:
    """Tests for Gemini chat completions"""

    def test_gemini_basic_chat(self, llm, system_message, simple_message, simple_format):
        """Test basic Gemini chat completion with structured output"""
        response = llm.ask(
            system_message=system_message,
            ids=["0"],
            user_messages=simple_message,
            response_formats=simple_format,
            model="gemini-2.0-flash",
            n_workers=1,
        )

        assert "0" in response
        assert "answer" in response["0"]
        assert "4" in response["0"]["answer"]

    def test_gemini_structured_output(self, llm, system_message, structured_response_format):
        """Test Gemini with complex structured output"""
        response = llm.ask(
            system_message=system_message,
            ids=["0"],
            user_messages=["Explain gravity in one sentence for each audience."],
            response_formats=structured_response_format,
            model="gemini-2.0-flash",
            n_workers=1,
        )

        assert "0" in response
        assert "FIVE_YEAR_OLD_EXPLANATION" in response["0"]
        assert "COLLEGE_STUDENT_EXPLANATION" in response["0"]


class TestGeminiErrorHandling:
    """Tests for Gemini error handling"""

    def test_gemini_bad_model_error(self, llm, system_message, simple_message, simple_format):
        """Test that invalid model errors are handled"""
        response = llm.ask(
            system_message=system_message,
            ids=["0"],
            user_messages=simple_message,
            response_formats=simple_format,
            model="gemini-nonexistent-model",
            n_workers=1,
        )

        assert "0" in response
        assert "error" in response["0"]


# ============== Concurrent Processing Tests ==============

class TestConcurrentProcessing:
    """Tests for concurrent chat processing"""

    def test_multiple_messages_concurrent(self, llm, system_message, simple_format):
        """Test processing multiple messages concurrently"""
        messages = [
            "What is 1+1? Answer with just the number.",
            "What is 2+2? Answer with just the number.",
            "What is 3+3? Answer with just the number.",
        ]
        formats = [simple_format[0]] * 3

        response = llm.ask(
            system_message=system_message,
            ids=["0", "1", "2"],
            user_messages=messages,
            response_formats=formats,
            model="gpt-4o-mini",
            n_workers=3,
        )

        assert len(response) == 3
        assert "0" in response
        assert "1" in response
        assert "2" in response


# ============== Error Response Format Tests ==============

class TestErrorResponseFormat:
    """Tests to verify error responses have consistent format"""

    def test_error_response_has_error_key(self, llm, system_message, simple_message, simple_format):
        """Test that error responses contain 'error' key"""
        response = llm.ask(
            system_message=system_message,
            ids=["0"],
            user_messages=simple_message,
            response_formats=simple_format,
            model="gpt-invalid-model-12345",
            n_workers=1,
        )

        if "error" in response["0"]:
            assert isinstance(response["0"]["error"], str)
            assert len(response["0"]["error"]) > 0

    def test_error_response_has_error_type(self, llm, system_message, simple_message, simple_format):
        """Test that error responses contain 'error_type' key"""
        response = llm.ask(
            system_message=system_message,
            ids=["0"],
            user_messages=simple_message,
            response_formats=simple_format,
            model="gpt-invalid-model-12345",
            n_workers=1,
        )

        if "error" in response["0"]:
            assert "error_type" in response["0"]
            assert response["0"]["error_type"] in ["auth", "bad_request", "max_retries", "parse_error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
