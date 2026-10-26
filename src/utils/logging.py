"""Small stderr helpers for application diagnostics."""

import sys


def log_error(message: str) -> None:
    """Report an application error on stderr."""
    print(f"[call_me_maybe] error: {message}", file=sys.stderr)


def log_warning(message: str) -> None:
    """Report a recoverable condition on stderr."""
    print(f"[call_me_maybe] warning: {message}", file=sys.stderr)
