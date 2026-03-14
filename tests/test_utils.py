"""Tests for utility functions."""

import pytest
from src.utils import extract_code_from_completion


class TestExtractCode:
    def test_python_fence(self):
        text = 'Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nDone.'
        code = extract_code_from_completion(text)
        assert code == "def add(a, b):\n    return a + b"

    def test_generic_fence(self):
        text = '```\ndef add(a, b):\n    return a + b\n```'
        code = extract_code_from_completion(text)
        assert code == "def add(a, b):\n    return a + b"

    def test_no_fence(self):
        text = "def add(a, b):\n    return a + b"
        code = extract_code_from_completion(text)
        assert code == "def add(a, b):\n    return a + b"

    def test_empty_string(self):
        assert extract_code_from_completion("") is None
        assert extract_code_from_completion("   ") is None

    def test_none_input(self):
        assert extract_code_from_completion(None) is None

    def test_conversational_format(self):
        messages = [
            {"role": "assistant", "content": '```python\ndef f(): return 1\n```'}
        ]
        code = extract_code_from_completion(messages)
        assert code == "def f(): return 1"

    def test_multiple_fences_takes_first(self):
        text = '```python\ndef first(): pass\n```\n\n```python\ndef second(): pass\n```'
        code = extract_code_from_completion(text)
        assert "first" in code

    def test_conversational_no_assistant(self):
        messages = [{"role": "user", "content": "hello"}]
        code = extract_code_from_completion(messages)
        assert code is None
