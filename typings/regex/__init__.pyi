"""Minimal typing surface used from the installed ``regex`` module."""

from collections.abc import Iterator
from re import Match


class Pattern:
    def finditer(self, string: str) -> Iterator[Match[str]]: ...


def compile(pattern: str) -> Pattern: ...
