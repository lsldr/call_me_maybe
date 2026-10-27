"""Bounded, model-free run trace recording and reading."""

from src.diagnostics.trace_io import TraceReader, TraceWriter, convert_legacy
from src.diagnostics.trace_schema import StateSnapshot, TraceRecord

__all__ = ["StateSnapshot", "TraceReader",
           "TraceRecord", "TraceWriter", "convert_legacy"]
