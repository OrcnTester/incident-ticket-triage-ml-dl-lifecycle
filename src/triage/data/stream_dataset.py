from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple


@dataclass(frozen=True)
class TicketRow:
    text: str
    priority: Optional[str] = None
    category: Optional[str] = None
    timestamp: Optional[str] = None
    system: Optional[str] = None
    source: Optional[str] = None
    error_code: Optional[str] = None
    routing_team: Optional[str] = None


def _row_to_ticket(row: Dict[str, str], *, text_col: str) -> TicketRow:
    def g(k: str) -> Optional[str]:
        v = row.get(k)
        if v is None:
            return None
        v = v.strip()
        return v if v else None

    return TicketRow(
        text=(row.get(text_col) or "").strip(),
        priority=g("priority"),
        category=g("category"),
        timestamp=g("timestamp"),
        system=g("system"),
        source=g("source"),
        error_code=g("error_code"),
        routing_team=g("routing_team"),
    )


class CSVTicketsStream:
    """
    A lightweight streaming dataset for incident tickets.

    - Uses csv.DictReader (constant memory).
    - Optional shuffle buffer to approximate shuffling in streaming mode.

    This is intentionally framework-agnostic (not tied to PyTorch/tf.data),
    but can be wrapped by those frameworks easily.
    """

    def __init__(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        text_col: str = "text",
        shuffle_buffer: int = 0,
        seed: int = 42,
    ) -> None:
        self.path = path
        self.encoding = encoding
        self.text_col = text_col
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[TicketRow]:
        if self.shuffle_buffer <= 0:
            yield from self._iter_plain()
        else:
            yield from self._iter_shuffle_buffer()

    def _iter_plain(self) -> Iterator[TicketRow]:
        with open(self.path, newline="", encoding=self.encoding) as f:
            r = csv.DictReader(f)
            for row in r:
                yield _row_to_ticket(row, text_col=self.text_col)

    def _iter_shuffle_buffer(self) -> Iterator[TicketRow]:
        """
        Buffered shuffle: keeps a window in memory and emits random items.
        Deterministic with a fixed seed for a single pass.
        """
        rng = random.Random(self.seed)
        buf: list[TicketRow] = []

        with open(self.path, newline="", encoding=self.encoding) as f:
            r = csv.DictReader(f)
            for row in r:
                buf.append(_row_to_ticket(row, text_col=self.text_col))
                if len(buf) >= self.shuffle_buffer:
                    i = rng.randrange(len(buf))
                    yield buf.pop(i)

        # drain remaining
        while buf:
            i = rng.randrange(len(buf))
            yield buf.pop(i)


def iter_text_and_label(
    path: str,
    *,
    target: str = "priority",
    shuffle_buffer: int = 0,
    seed: int = 42,
) -> Iterator[Tuple[str, Optional[str]]]:
    """
    Convenience iterator that yields (text, label) for a target.
    target: "priority" or "category"
    """
    ds = CSVTicketsStream(path, shuffle_buffer=shuffle_buffer, seed=seed)
    for t in ds:
        label = t.priority if target == "priority" else t.category
        yield t.text, label
