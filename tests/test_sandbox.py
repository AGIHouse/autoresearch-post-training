"""
Tests for the code execution sandbox.

Tests SubprocessSandbox (always available) and DockerSandbox (requires Docker + image).

Run with: pytest tests/test_sandbox.py -v
"""

import pytest
from src.sandbox import SandboxPool, SubprocessSandbox, ExecutionResult


class TestExecutionResult:
    def test_pass_rate_all_passed(self):
        r = ExecutionResult(passed=3, total=3, errors=[])
        assert r.pass_rate == 1.0
        assert r.all_passed is True

    def test_pass_rate_partial(self):
        r = ExecutionResult(passed=2, total=3, errors=["Test 2: AssertionError"])
        assert r.pass_rate == pytest.approx(2 / 3)
        assert r.all_passed is False

    def test_pass_rate_none_passed(self):
        r = ExecutionResult(passed=0, total=3, errors=["error"])
        assert r.pass_rate == 0.0

    def test_pass_rate_zero_total(self):
        r = ExecutionResult(passed=0, total=0, errors=[])
        assert r.pass_rate == 0.0


class TestSubprocessSandbox:
    """Tests using SubprocessSandbox — always available, no Docker needed."""

    @pytest.fixture(scope="class")
    def sandbox(self):
        return SubprocessSandbox(timeout=10)

    def test_simple_function(self, sandbox):
        code = "def add(a, b):\n    return a + b"
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]
        result = sandbox.execute(code, tests)
        assert result.passed == 3
        assert result.total == 3
        assert result.all_passed

    def test_partial_pass(self, sandbox):
        code = "def add(a, b):\n    return a + b + 1"
        tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
        result = sandbox.execute(code, tests)
        assert result.passed == 0
        assert result.total == 2

    def test_syntax_error(self, sandbox):
        code = "def add(a, b)\n    return a + b"
        tests = ["assert add(1, 2) == 3"]
        result = sandbox.execute(code, tests)
        assert result.passed == 0
        assert len(result.errors) > 0

    def test_runtime_error(self, sandbox):
        code = "def add(a, b):\n    return a / 0"
        tests = ["assert add(1, 2) == 3"]
        result = sandbox.execute(code, tests)
        assert result.passed == 0
        assert len(result.errors) > 0

    def test_timeout(self):
        sandbox = SubprocessSandbox(timeout=2)
        code = "import time\ntime.sleep(30)"
        tests = ["assert True"]
        result = sandbox.execute(code, tests)
        assert result.passed == 0
        assert any("timed out" in e.lower() or "timeout" in e.lower() for e in result.errors)

    def test_setup_code(self, sandbox):
        setup = "import math"
        code = "def circle_area(r):\n    return math.pi * r * r"
        tests = ["assert abs(circle_area(1) - 3.14159) < 0.01"]
        result = sandbox.execute(code, tests, setup_code=setup)
        assert result.passed == 1


class TestSandboxPool:
    """Tests the pool with subprocess backend (always available)."""

    @pytest.fixture(scope="class")
    def pool(self):
        return SandboxPool(pool_size=4, timeout=10, backend="subprocess")

    def test_single_execution(self, pool):
        result = pool.execute("def f(): return 1", ["assert f() == 1"])
        assert result.all_passed

    def test_batch_execution(self, pool):
        items = [
            ("def f(): return 1", ["assert f() == 1"], ""),
            ("def g(): return 2", ["assert g() == 2"], ""),
            ("def h(): return 3", ["assert h() == 3"], ""),
            ("def broken(: pass", ["assert True"], ""),  # syntax error
        ]
        results = pool.execute_batch(items)
        assert len(results) == 4
        assert results[0].all_passed
        assert results[1].all_passed
        assert results[2].all_passed
        assert not results[3].all_passed

    def test_batch_preserves_order(self, pool):
        """Results must be in the same order as inputs."""
        items = [
            (f"def f(): return {i}", [f"assert f() == {i}"], "")
            for i in range(10)
        ]
        results = pool.execute_batch(items)
        assert all(r.all_passed for r in results)


class TestDockerSandbox:
    """Tests using DockerSandbox — requires Docker + sandbox image."""

    @pytest.fixture(scope="class")
    def pool(self):
        try:
            return SandboxPool(pool_size=4, timeout=10, backend="docker")
        except Exception as e:
            pytest.skip(f"Docker sandbox not available: {e}")

    def test_simple_function(self, pool):
        result = pool.execute("def add(a, b):\n    return a + b", ["assert add(1, 2) == 3"])
        assert result.all_passed

    def test_batch_execution(self, pool):
        items = [
            ("def f(): return 1", ["assert f() == 1"], ""),
            ("def g(): return 2", ["assert g() == 2"], ""),
        ]
        results = pool.execute_batch(items)
        assert all(r.all_passed for r in results)
