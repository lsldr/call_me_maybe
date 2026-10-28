"""Bounded streaming JSONL trace writer and indexed reader."""

import json
from pathlib import Path
from typing import Any, Callable, Iterator

from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr

from src.diagnostics.trace_schema import TraceRecord


class TraceWriter(BaseModel):
    """Write bounded trace records directly to a JSONL file."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    path: Path = Field(description="Destination JSONL path")
    max_record_bytes: int = Field(
        default=65_536, ge=1, description="Maximum encoded event size")
    max_file_bytes: int = Field(
        default=67_108_864, ge=1, description="Maximum trace file size")
    max_events: int = Field(
        default=100_000, ge=1, description="Maximum events including run_end")
    observer: Callable[[TraceRecord], None] | None = None
    _file: Any = PrivateAttr(default=None)
    _seq: int = PrivateAttr(default=0)
    _written: int = PrivateAttr(default=0)
    _closed: bool = PrivateAttr(default=True)
    _truncated: bool = PrivateAttr(default=False)
    _status: JsonValue = PrivateAttr(default=None)
    _observed: int = PrivateAttr(default=0)

    @property
    def is_open(self) -> bool:
        """Return whether a terminal record can still be written."""
        return not self._closed

    def open(self) -> "TraceWriter":
        """Preflight and open the destination, replacing it intentionally."""
        if self.is_open:
            raise ValueError("trace writer is already open")
        if not self.path.parent.exists() or not self.path.parent.is_dir():
            raise FileNotFoundError(
                f"trace parent directory does not exist: {self.path.parent}")
        reserve = self._end_reserve()
        if reserve > self.max_record_bytes or reserve > self.max_file_bytes:
            raise ValueError(
                "trace limits cannot fit a terminal run_end record")
        # This is the explicit point at which an old trace is replaced.
        self._file = self.path.open("wb")
        self._seq = self._written = 0
        self._closed = False
        self._truncated = False
        self._status = None
        self._observed = 0
        return self

    def emit(
        self, kind: str, data: dict[str, JsonValue] | None = None
    ) -> TraceRecord | None:
        """Append one event, returning None after a soft trace limit."""
        if self._closed:
            return None
        if kind == "run_end":
            raise ValueError("use finish() to write run_end")
        record = TraceRecord.model_validate({
            "kind": kind, "seq": self._seq, "data": data or {}
        })
        if self.observer is not None:
            self.observer(record.model_copy(update={"seq": self._observed}))
            self._observed += 1
        if self._truncated:
            return None
        encoded = _encode(record)
        if (len(encoded) > self.max_record_bytes
                or self._written + len(encoded) + self._end_reserve()
                > self.max_file_bytes
                or self._seq + 1 >= self.max_events):
            self._truncated = True
            self._status = (data or {}).get("status")
            return None
        self._write(encoded)
        self._seq += 1
        return record

    def finish(self, data: dict[str, JsonValue] | None = None) -> TraceRecord:
        """Append the reserved run_end event and close the file."""
        if self._closed:
            raise ValueError("trace writer is not open")
        ending = dict(data or {})
        if self._truncated:
            ending["truncated"] = True
            ending["original_status"] = ending.get("status", self._status)
        record = TraceRecord(kind="run_end", seq=self._seq, data=ending)
        encoded = _encode(record)
        if (len(encoded) > self.max_record_bytes or
                self._written + len(encoded) > self.max_file_bytes):
            # A compact terminal event preserves the bounded trace contract.
            original_status = ending.get("status", self._status)
            footer_data: dict[str, JsonValue] = {
                "truncated": True,
                "original_status": _bounded_status(original_status),
            }
            if _bounded_status(original_status) != original_status:
                footer_data["original_status_truncated"] = True
            record = TraceRecord(
                kind="run_end", seq=self._seq,
                data=footer_data
            )
            encoded = _encode(record)
        if (len(encoded) > self.max_record_bytes
                or self._written + len(encoded) > self.max_file_bytes):
            self.close()
            raise OSError("max_file_bytes is too small to store run_end")
        self._write(encoded)
        self._seq += 1
        if self.observer is not None:
            self.observer(record.model_copy(update={"seq": self._observed}))
            self._observed += 1
        self.close()
        return record

    def close(self) -> None:
        """Close an open trace file without writing a terminal event."""
        file = self._file
        self._file = None
        self._closed = True
        if file is not None:
            file.close()

    def _end_reserve(self) -> int:
        """Return the maximum encoded space reserved for a terminal record.

        Returns:
            Byte length reserved for the bounded ``run_end`` record.
        """
        reserve_record = TraceRecord(
            kind="run_end", seq=self._seq + 1,
            data={"truncated": True, "original_status": "\\" * 64,
                  "original_status_truncated": True}
        )
        return len(_encode(reserve_record))

    def _write(self, encoded: bytes) -> None:
        """Write a complete encoded record, closing the file on I/O failure.

        Args:
            encoded: Newline-terminated JSON record bytes.

        Raises:
            OSError: If the file cannot accept the complete record.
        """
        try:
            assert self._file is not None
            written = self._file.write(encoded)
            if written != len(encoded):
                raise OSError("short write while recording trace")
            self._written += len(encoded)
        except OSError:
            try:
                self.close()
            except OSError:
                pass
            raise


class TraceReader(BaseModel):
    """Validate a bounded JSONL trace and retain offsets for indexed reads."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    path: Path = Field(description="JSONL trace to index")
    max_record_bytes: int = Field(
        default=65_536, ge=1, description="Maximum encoded event size")
    max_file_bytes: int = Field(
        default=67_108_864, ge=1, description="Maximum trace file size")
    max_events: int = Field(
        default=100_000, ge=1, description="Maximum records to index")
    allow_partial: bool = Field(
        default=False, description="Allow an incomplete final JSONL line")
    _offsets: list[tuple[int, int]] = PrivateAttr(default_factory=list)
    _partial: bool = PrivateAttr(default=False)

    @property
    def partial(self) -> bool:
        """Return whether the trace is incomplete or truncated."""
        return self._partial

    def open(self) -> "TraceReader":
        """Scan and validate the file while storing offsets only."""
        size = self.path.stat().st_size
        if size > self.max_file_bytes:
            raise ValueError("trace exceeds max_file_bytes")
        self._offsets.clear()
        self._partial = False
        previous = -1
        last_record: TraceRecord | None = None
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline(self.max_record_bytes + 2)
                if not line:
                    break
                if (len(line) > self.max_record_bytes
                        and not line.endswith(b"\n")):
                    raise ValueError("trace record exceeds max_record_bytes")
                if not line.endswith(b"\n"):
                    if self.allow_partial:
                        self._partial = True
                        break
                    raise ValueError("partial final trace record")
                if len(line) > self.max_record_bytes:
                    raise ValueError("trace record exceeds max_record_bytes")
                record = _decode(line)
                if record.seq <= previous:
                    raise ValueError(
                        "trace seq values must be strictly increasing")
                previous = record.seq
                last_record = record
                self._offsets.append((offset, len(line)))
                if len(self._offsets) > self.max_events:
                    raise ValueError("trace exceeds max_events")
        if (not self._partial and
                (last_record is None or last_record.kind != "run_end"
                 or last_record.data.get("truncated") is True)):
            self._partial = True
        return self

    def __len__(self) -> int:
        """Return the number of complete records in the trace."""
        return len(self._offsets)

    def read(self, index: int) -> TraceRecord:
        """Read and validate one indexed event."""
        offset, length = self._offsets[index]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return _decode(handle.read(length))

    def __getitem__(self, index: int) -> TraceRecord:
        """Read one record by its zero-based index."""
        return self.read(index)

    def __iter__(self) -> Iterator[TraceRecord]:  # type: ignore[override]
        """Yield complete records in file order."""
        for index in range(len(self)):
            yield self.read(index)


def convert_legacy(
    records: list[dict[str, JsonValue]]
) -> list[TraceRecord]:
    """Convert old tokens with explicit source fields and no repr parsing."""
    converted: list[TraceRecord] = []
    for seq, old in enumerate(records):
        raw = old.get("raw_hex", old.get("raw"))
        step = old.get("step")
        token_id = old.get("token_id")
        allowed_count = old.get("allowed_count")
        byte_count = old.get("byte_count")
        complete = old.get("complete")
        if (not _legacy_nonnegative_int(step)
                or not _legacy_nonnegative_int(token_id)
                or not _legacy_nonnegative_int(byte_count)
                or not _legacy_nonnegative_int(allowed_count)
                or allowed_count == 0
                or not isinstance(complete, bool)
                or not isinstance(raw, str)):
            raise ValueError("legacy token is missing required token fields")
        raw_hex = raw
        try:
            bytes.fromhex(raw_hex)
        except ValueError:
            raise ValueError("legacy token raw bytes are not valid hex")
        prompt = old.get("prompt_index")
        case_id = prompt if isinstance(prompt, str) else old.get("case_id")
        data: dict[str, JsonValue] = {
            "step": step,
            "token_id": token_id,
            "raw_hex": raw_hex,
            "allowed_count": allowed_count,
            "byte_count": byte_count,
            "complete": complete,
            "provenance": "legacy",
            "prompt_index": (
                prompt if isinstance(prompt, int)
                and not isinstance(prompt, bool) and prompt >= 0 else None
            ),
            "case_id": case_id if isinstance(case_id, str) else None,
            "elapsed_seconds": None,
        }
        if isinstance(old.get("state"), str):
            data["legacy_state"] = old["state"]
        converted.append(TraceRecord(kind="token", seq=seq, data=data))
    return converted


def _encode(record: TraceRecord) -> bytes:
    """Serialize a validated record as one UTF-8 JSONL line.

    Args:
        record: Validated trace record to encode.

    Returns:
        JSON bytes terminated by a newline.
    """
    return (record.model_dump_json() + "\n").encode("utf-8")


def _decode(line: bytes) -> TraceRecord:
    """Parse and validate one JSONL record.

    Args:
        line: UTF-8 JSON object bytes, optionally newline-terminated.

    Returns:
        Validated trace record.

    Raises:
        ValueError: If the bytes are invalid JSON or violate the schema.
    """
    try:
        value = json.loads(line, parse_constant=lambda value: (
            _ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid JSON trace record") from exc
    if not isinstance(value, dict):
        raise ValueError("trace record must be a JSON object")
    return TraceRecord.model_validate(value)


def _legacy_nonnegative_int(value: JsonValue) -> bool:
    """Check whether a legacy JSON value is a nonnegative non-boolean int."""
    return (isinstance(value, int) and not isinstance(value, bool)
            and value >= 0)


def _bounded_status(value: JsonValue) -> JsonValue:
    """Sanitize and cap a status string for compact terminal metadata.

    Args:
        value: Candidate JSON status value.

    Returns:
        A printable ASCII string of at most 64 characters, or ``None``.
    """
    if isinstance(value, str):
        return "".join(
            char if char.isascii() and char.isprintable() else "?"
            for char in value[:64]
        )
    return None
