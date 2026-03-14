"""
Code execution sandbox for safe execution during RL training.

Two backends are supported:

1. **DockerSandbox** (production):
   Each execution runs inside an ephemeral Docker container with:
   - No network access (--network none)
   - Limited memory (256MB default)
   - Limited PIDs (64, prevents fork bombs)
   - CPU throttling (50% of one core)
   - Wall-clock timeout

2. **SubprocessSandbox** (development/testing):
   Uses subprocess with resource limits. Faster but less isolated.
   Suitable for local development and CI where Docker isn't available.

Both backends implement the same interface and return ExecutionResult.

The SandboxPool dispatches to either backend and uses ThreadPoolExecutor
for parallel execution, matching the GRPO batch size.

Architecture references:
- EvalPlus uses Docker with configurable resource limits
- Open-R1 uses E2B Firecracker microVMs (stronger isolation)
- DeepSWE uses 500+ concurrent Docker containers on Kubernetes
- SWE-bench uses stateless subprocess.run (trivially sandboxable)
"""

import json
import logging
import subprocess
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing code against test cases in the sandbox."""
    passed: int
    total: int
    errors: list[str]

    @property
    def pass_rate(self) -> float:
        """Fraction of tests passed (0.0 to 1.0)."""
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def all_passed(self) -> bool:
        return self.passed == self.total and self.total > 0


class SubprocessSandbox:
    """
    Lightweight sandbox using subprocess for local development.

    Uses subprocess with timeout. Less isolated than Docker but works
    anywhere Python is installed. Suitable for:
    - Local development and debugging
    - CI environments without Docker
    - Quick iteration on reward function logic

    NOT suitable for production training with untrusted model outputs.
    Use DockerSandbox for that.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def execute(
        self,
        code: str,
        tests: list[str],
        setup_code: str = "",
    ) -> ExecutionResult:
        """Execute code + tests in a subprocess."""
        # Build a self-contained test script
        script = self._build_script(code, tests, setup_code)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=True
            ) as f:
                f.write(script)
                f.flush()

                result = subprocess.run(
                    ["python3", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

            # Parse JSON output from the script
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip().split("\n")[-1])
                return ExecutionResult(
                    passed=data.get("passed", 0),
                    total=data.get("total", len(tests)),
                    errors=data.get("errors", []),
                )
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return ExecutionResult(passed=0, total=len(tests), errors=[error_msg])

        except subprocess.TimeoutExpired:
            return ExecutionResult(passed=0, total=len(tests), errors=["Execution timed out"])
        except json.JSONDecodeError as e:
            return ExecutionResult(passed=0, total=len(tests), errors=[f"Output parse error: {e}"])
        except Exception as e:
            return ExecutionResult(passed=0, total=len(tests), errors=[str(e)])

    def _build_script(self, code: str, tests: list[str], setup_code: str) -> str:
        """Build a Python script that runs code + tests and outputs JSON."""
        # Escape the code and tests for embedding in the script
        return textwrap.dedent(f"""\
            import json
            import sys

            results = {{"passed": 0, "total": {len(tests)}, "errors": []}}
            namespace = {{}}

            try:
                # Setup code
                exec({repr(setup_code)}, namespace)
                # Solution code
                exec({repr(code)}, namespace)
                # Run tests
                tests = {repr(tests)}
                for i, test in enumerate(tests):
                    try:
                        exec(test, namespace)
                        results["passed"] += 1
                    except AssertionError as e:
                        results["errors"].append(f"Test {{i}}: AssertionError: {{e}}")
                    except Exception as e:
                        results["errors"].append(f"Test {{i}}: {{type(e).__name__}}: {{e}}")
            except SyntaxError as e:
                results["errors"].append(f"SyntaxError: {{e}}")
            except Exception as e:
                results["errors"].append(f"{{type(e).__name__}}: {{e}}")

            print(json.dumps(results))
        """)


class DockerSandbox:
    """
    Production sandbox using Docker containers via subprocess.

    Shells out to `docker run` via subprocess.run() instead of the Docker
    Python SDK. This is the pattern used by EvalPlus and other production
    systems because subprocess.run(timeout=N) provides reliable OS-level
    timeout enforcement — unlike the Docker SDK's container.wait() which
    can hang indefinitely on certain workloads.

    Each execution runs in an ephemeral container (--rm) with:
    - No network access (--network none)
    - Limited memory (--memory 256m)
    - Limited PIDs (--pids-limit 64, prevents fork bombs)
    - CPU throttling (--cpus 0.5)
    - Wall-clock timeout via subprocess.run()
    """

    def __init__(
        self,
        timeout: int = 10,
        memory_limit: str = "256m",
        image: str = "coding-sandbox:latest",
    ):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.image = image

        # Verify docker is available and image exists
        result = subprocess.run(
            ["docker", "image", "inspect", self.image],
            capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Sandbox image '{self.image}' not found. "
                f"Run 'make sandbox' to build it."
            )

    def execute(
        self,
        code: str,
        tests: list[str],
        setup_code: str = "",
    ) -> ExecutionResult:
        """Execute code + tests in a Docker container via subprocess."""
        payload = json.dumps({
            "code": code,
            "tests": tests,
            "setup_code": setup_code,
            "timeout": self.timeout,
        })

        # Build docker run command with resource limits.
        # Pass payload via stdin (-i flag) rather than env var to avoid
        # size limits on environment variables.
        cmd = [
            "docker", "run", "--rm", "-i",
            "--network", "none",
            "--memory", self.memory_limit,
            "--pids-limit", "64",
            "--cpus", "0.5",
            self.image,
        ]

        try:
            result = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout + 5,  # Hard OS-level timeout
            )

            if result.stdout.strip():
                parsed = json.loads(result.stdout.strip().split("\n")[-1])
                return ExecutionResult(
                    passed=parsed.get("passed", 0),
                    total=parsed.get("total", len(tests)),
                    errors=parsed.get("errors", []),
                )
            else:
                error_msg = result.stderr.strip()[:500] if result.stderr else "No output"
                return ExecutionResult(passed=0, total=len(tests), errors=[error_msg])

        except subprocess.TimeoutExpired:
            return ExecutionResult(passed=0, total=len(tests), errors=["Execution timed out"])
        except json.JSONDecodeError as e:
            return ExecutionResult(passed=0, total=len(tests), errors=[f"Output parse error: {e}"])
        except Exception as e:
            logger.debug(f"Sandbox error: {type(e).__name__}: {e}")
            return ExecutionResult(passed=0, total=len(tests), errors=[str(e)])


class SandboxPool:
    """
    Pool for parallel code execution with automatic backend selection.

    Tries Docker first (production). Falls back to subprocess (development).
    Uses ThreadPoolExecutor for parallel execution.

    Usage:
        pool = SandboxPool(pool_size=8)
        results = pool.execute_batch([
            ("def add(a,b): return a+b", ["assert add(1,2)==3"], ""),
            ("def mul(a,b): return a*b", ["assert mul(2,3)==6"], ""),
        ])
    """

    def __init__(
        self,
        pool_size: int = 8,
        timeout: int = 10,
        memory_limit: str = "256m",
        image: str = "coding-sandbox:latest",
        backend: str = "auto",
    ):
        self.pool_size = pool_size
        self.backend_name = backend

        if backend == "auto":
            try:
                self._backend = DockerSandbox(timeout, memory_limit, image)
                self.backend_name = "docker"
                logger.info("Sandbox backend: Docker")
            except Exception as e:
                logger.warning(f"Docker unavailable ({e}), falling back to subprocess")
                self._backend = SubprocessSandbox(timeout)
                self.backend_name = "subprocess"
        elif backend == "docker":
            self._backend = DockerSandbox(timeout, memory_limit, image)
            logger.info("Sandbox backend: Docker")
        elif backend == "subprocess":
            self._backend = SubprocessSandbox(timeout)
            logger.info("Sandbox backend: subprocess")
        else:
            raise ValueError(f"Unknown sandbox backend: {backend}")

    def execute(
        self,
        code: str,
        tests: list[str],
        setup_code: str = "",
    ) -> ExecutionResult:
        """Execute code + tests using the selected backend."""
        return self._backend.execute(code, tests, setup_code)

    def execute_batch(
        self,
        items: list[tuple[str, list[str], str]],
    ) -> list[ExecutionResult]:
        """
        Execute multiple (code, tests, setup_code) tuples in parallel.

        Results are returned in the same order as inputs.
        """
        results: list[ExecutionResult | None] = [None] * len(items)
        # Per-future timeout: 3x the sandbox timeout
        per_future_timeout = self._backend.timeout * 3
        # Total timeout: account for serial execution with pool_size workers
        import math
        total_timeout = math.ceil(len(items) / self.pool_size) * per_future_timeout + 60

        with ThreadPoolExecutor(max_workers=self.pool_size) as executor:
            future_to_idx = {
                executor.submit(self.execute, code, tests, setup): idx
                for idx, (code, tests, setup) in enumerate(items)
            }

            for future in as_completed(future_to_idx, timeout=total_timeout):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=per_future_timeout)
                except Exception as e:
                    logger.error(f"Batch execution error at index {idx}: {e}")
                    results[idx] = ExecutionResult(
                        passed=0,
                        total=len(items[idx][1]),
                        errors=[str(e)],
                    )

        # Fill any remaining None results (from timeout)
        for i, r in enumerate(results):
            if r is None:
                results[i] = ExecutionResult(
                    passed=0, total=len(items[i][1]), errors=["Execution timed out"]
                )

        return results  # type: ignore[return-value]
