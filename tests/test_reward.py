"""
Tests for the reward functions.

TestFormatReward: tests format_reward without Docker.
TestCodeExecutionReward: tests code_execution_reward with a mocked sandbox pool.
TestBinaryRewardMode: tests the binary (DeepSWE-style) reward mode.

Requires the `docker` Python package to be installed (since reward.py imports sandbox.py).
On environments without it, tests are skipped gracefully.
"""

from unittest.mock import MagicMock
import pytest

from src.reward import format_reward, code_execution_reward, set_sandbox_pool, set_reward_mode
from src.sandbox import ExecutionResult


class TestFormatReward:
    def test_proper_python_block(self):
        completion = '```python\ndef add(a, b):\n    return a + b\n```'
        rewards = format_reward(["prompt"], [completion])
        assert rewards == [1.0]

    def test_no_code_block(self):
        completion = "def add(a, b):\n    return a + b"
        rewards = format_reward(["prompt"], [completion])
        assert rewards == [0.0]

    def test_generic_code_block(self):
        completion = '```\ndef add(a, b):\n    return a + b\n```'
        rewards = format_reward(["prompt"], [completion])
        assert rewards == [0.0]

    def test_batch(self):
        completions = [
            '```python\ndef f(): pass\n```',
            "def f(): pass",
            '```python\ndef g(): return 1\n```',
        ]
        rewards = format_reward(["p"] * 3, completions)
        assert rewards == [1.0, 0.0, 1.0]

    def test_conversational_format(self):
        completion = [{"role": "assistant", "content": '```python\ndef f(): pass\n```'}]
        rewards = format_reward(["prompt"], [completion])
        assert rewards == [1.0]

    def test_no_newline_before_closing(self):
        completion = '```python\ndef add(a, b): return a + b```'
        rewards = format_reward(["prompt"], [completion])
        assert rewards == [1.0]


class TestCodeExecutionReward:
    @pytest.fixture(autouse=True)
    def setup_mock_pool(self):
        self.mock_pool = MagicMock()
        set_sandbox_pool(self.mock_pool)
        set_reward_mode("partial")
        yield
        set_sandbox_pool(None)

    def test_all_tests_pass(self):
        self.mock_pool.execute_batch.return_value = [
            ExecutionResult(passed=3, total=3, errors=[]),
        ]
        rewards = code_execution_reward(
            prompts=["write add"],
            completions=['```python\ndef add(a,b): return a+b\n```'],
            test_list=[["assert add(1,2)==3", "assert add(0,0)==0", "assert add(-1,1)==0"]],
            test_setup_code=[""],
        )
        assert rewards == [1.0]

    def test_partial_credit(self):
        self.mock_pool.execute_batch.return_value = [
            ExecutionResult(passed=2, total=3, errors=["Test 2: AssertionError"]),
        ]
        rewards = code_execution_reward(
            prompts=["write add"],
            completions=['```python\ndef add(a,b): return a+b+1\n```'],
            test_list=[["t1", "t2", "t3"]],
            test_setup_code=[""],
        )
        assert rewards == [pytest.approx(2 / 3)]

    def test_error_penalty(self):
        self.mock_pool.execute_batch.return_value = [
            ExecutionResult(passed=0, total=3, errors=["SyntaxError"]),
        ]
        rewards = code_execution_reward(
            prompts=["write add"],
            completions=['```python\ndef add(a,b)\n```'],
            test_list=[["t1", "t2", "t3"]],
            test_setup_code=[""],
        )
        assert rewards == [-0.5]

    def test_no_code_extracted(self):
        rewards = code_execution_reward(
            prompts=["write add"],
            completions=[""],
            test_list=[["t1"]],
            test_setup_code=[""],
        )
        self.mock_pool.execute_batch.assert_not_called()
        assert rewards == [-0.5]

    def test_batch_mixed_results(self):
        self.mock_pool.execute_batch.return_value = [
            ExecutionResult(passed=3, total=3, errors=[]),
            ExecutionResult(passed=0, total=3, errors=["err"]),
        ]
        rewards = code_execution_reward(
            prompts=["p1", "p2", "p3"],
            completions=[
                '```python\ndef f(): return 1\n```',
                "",
                '```python\ndef h(): oops\n```',
            ],
            test_list=[["t1", "t2", "t3"]] * 3,
            test_setup_code=[""] * 3,
        )
        assert len(rewards) == 3
        assert rewards[0] == 1.0
        assert rewards[1] == -0.5
        assert rewards[2] == -0.5

    def test_none_test_list(self):
        rewards = code_execution_reward(
            prompts=["p"],
            completions=["code"],
            test_list=None,
        )
        assert rewards == [0.0]

    def test_zero_total_tests(self):
        self.mock_pool.execute_batch.return_value = [
            ExecutionResult(passed=0, total=0, errors=[]),
        ]
        rewards = code_execution_reward(
            prompts=["p"],
            completions=['```python\ndef f(): pass\n```'],
            test_list=[[]],
            test_setup_code=[""],
        )
        assert rewards == [0.0]


class TestBinaryRewardMode:
    """Test binary reward mode (DeepSWE-style: 1.0 if all pass, 0.0 otherwise)."""

    @pytest.fixture(autouse=True)
    def setup_mock_pool(self):
        self.mock_pool = MagicMock()
        set_sandbox_pool(self.mock_pool)
        set_reward_mode("binary")
        yield
        set_reward_mode("partial")
        set_sandbox_pool(None)

    def test_all_pass_binary(self):
        self.mock_pool.execute_batch.return_value = [
            ExecutionResult(passed=3, total=3, errors=[]),
        ]
        rewards = code_execution_reward(
            prompts=["p"],
            completions=['```python\ndef f(): return 1\n```'],
            test_list=[["t1", "t2", "t3"]],
            test_setup_code=[""],
        )
        assert rewards == [1.0]

    def test_partial_pass_binary_gives_zero(self):
        """In binary mode, passing 2/3 tests gives 0.0 (not 0.67)."""
        self.mock_pool.execute_batch.return_value = [
            ExecutionResult(passed=2, total=3, errors=[]),
        ]
        rewards = code_execution_reward(
            prompts=["p"],
            completions=['```python\ndef f(): return 1\n```'],
            test_list=[["t1", "t2", "t3"]],
            test_setup_code=[""],
        )
        assert rewards == [0.0]

    def test_error_still_penalized_binary(self):
        """Errors still get -0.5 in binary mode."""
        self.mock_pool.execute_batch.return_value = [
            ExecutionResult(passed=0, total=3, errors=["SyntaxError"]),
        ]
        rewards = code_execution_reward(
            prompts=["p"],
            completions=['```python\ndef f(:\n```'],
            test_list=[["t1", "t2", "t3"]],
            test_setup_code=[""],
        )
        assert rewards == [-0.5]
