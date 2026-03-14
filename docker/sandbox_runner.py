"""
Sandbox execution script that runs inside a Docker container.

Protocol:
  - Reads a JSON payload from stdin:
    {"code": "def foo(x): ...", "tests": ["assert foo(1) == 2"], "setup_code": "", "timeout": 5}
  - Executes the code + tests in an isolated namespace
  - Writes a JSON result to stdout:
    {"passed": 2, "total": 3, "errors": ["AssertionError on test 2"]}

Security: This script runs inside a Docker container with no network,
limited memory, limited PIDs, and a wall-clock timeout enforced by the
container runtime. The signal-based timeout here is an additional safeguard.
"""

import json
import signal
import sys
import traceback


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def run_code_with_tests(code: str, tests: list[str], setup_code: str = "", timeout: int = 5):
    """Execute code and run test assertions against it."""
    results = {"passed": 0, "total": len(tests), "errors": []}

    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        # Create isolated namespace
        namespace = {}

        # Run setup code if provided (e.g., helper imports)
        if setup_code:
            exec(setup_code, namespace)

        # Execute the solution code
        exec(code, namespace)

        # Run each test
        for i, test in enumerate(tests):
            try:
                exec(test, namespace)
                results["passed"] += 1
            except AssertionError as e:
                results["errors"].append(f"Test {i}: AssertionError: {e}")
            except Exception as e:
                results["errors"].append(f"Test {i}: {type(e).__name__}: {e}")

    except TimeoutError:
        results["errors"].append("Execution timed out")
    except SyntaxError as e:
        results["errors"].append(f"SyntaxError: {e}")
    except Exception as e:
        results["errors"].append(f"{type(e).__name__}: {e}")
    finally:
        signal.alarm(0)  # Cancel timeout

    return results


def main():
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        print(json.dumps({"passed": 0, "total": 0, "errors": [f"Invalid JSON input: {e}"]}))
        sys.exit(1)

    code = payload.get("code", "")
    tests = payload.get("tests", [])
    setup_code = payload.get("setup_code", "")
    timeout = payload.get("timeout", 5)

    result = run_code_with_tests(code, tests, setup_code, timeout)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
