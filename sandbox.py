"""
Subprocess-based Python code execution sandbox.

Runs each code snippet in a fresh subprocess with a hard timeout.
No Docker required — simpler and about 12% faster for development/small models.

Isolation guarantee: each call is a new process, so crashes/infinite-loops in
generated code cannot affect the training process. Memory isolation is OS-level
(separate address space). Network is not blocked — for training, that's fine;
for production eval, consider Docker.
"""

import ast
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class ExecResult:
    passed: int    # how many assert statements passed
    total: int     # total assert statements run
    error: str     # last error message (empty string if clean)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total


def run_code(code: str, tests: list[str], setup: str = "", timeout: float = 5.0) -> ExecResult:
    """
    Run Python code against a list of assert statements.

    Each assertion is run in its own subprocess call so we can count partial
    passes. This is slightly slower than one big subprocess, but gives a richer
    reward signal (passed=2/3 vs all-or-nothing).

    Args:
        code:    The generated Python code (function definitions etc.)
        tests:   List of assertion strings, e.g. ["assert foo(1) == 2", ...]
        setup:   Optional import/setup code to prepend (e.g. "import math")
        timeout: Per-test timeout in seconds

    Returns:
        ExecResult with passed count, total count, and last error message
    """
    # Quick syntax check — avoids subprocess overhead for obviously broken code
    try:
        ast.parse(code)
    except SyntaxError as e:
        return ExecResult(passed=0, total=len(tests), error=f"SyntaxError: {e}")

    if not tests:
        return ExecResult(passed=0, total=0, error="no tests")

    header = "\n".join(filter(None, [setup.strip(), code.strip()]))
    passed = 0
    last_error = ""

    for test in tests:
        script = f"{header}\n{test}\n"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode == 0:
                passed += 1
            else:
                last_error = (proc.stderr or proc.stdout).strip()[:300]
        except subprocess.TimeoutExpired:
            last_error = "TimeoutExpired"
            break
        except Exception as e:
            last_error = str(e)
            break

    return ExecResult(passed=passed, total=len(tests), error=last_error)


def run_batch(
    items: list[tuple[str, list[str], str]],
    workers: int = 8,
    timeout: float = 5.0,
) -> list[ExecResult]:
    """
    Execute a list of (code, tests, setup) triples in parallel.

    Returns results in the same order as the inputs.
    """
    results: list[ExecResult | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_code, code, tests, setup, timeout): i
            for i, (code, tests, setup) in enumerate(items)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = ExecResult(passed=0, total=0, error=str(e))
    return results  # type: ignore[return-value]
