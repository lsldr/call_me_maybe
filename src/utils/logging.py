"""Small stderr helpers for application diagnostics."""

import sys


def log_error(message: str) -> None:
    """Report an application error on stderr."""
    print(f"[call_me_maybe] error: {message}", file=sys.stderr)
