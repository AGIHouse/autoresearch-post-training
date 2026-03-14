"""
Sandbox execution script that runs inside a Docker container.

Protocol:
  - Reads JSON from stdin: {"code": "...", "tests": ["assert ..."], "setup_code": "", "timeout": 5}
  - Executes code + tests in an isolated namespace
  - Writes JSON to stdout: {"passed": 2, "total": 3, "errors": [...]}
"""

import json
import signal
import sys


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def run_code_with_tests(code: str, tests: list[str], setup_code: str = "", timeout: int = 5):
    """Execute code and run test assertions against it."""
    pass


def main():
    pass


if __name__ == "__main__":
    main()
