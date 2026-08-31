from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


class Timer:
    """Accumulates wall-clock time per named section.

    Used to reproduce the latency columns of the paper (Stage-1 filtering time
    vs. Stage-2 re-ranking time vs. end-to-end per-query cost).
    """

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def mean_ms(self, name: str) -> float:
        count = self.counts.get(name, 0)
        return 1000.0 * self.totals.get(name, 0.0) / count if count else 0.0

    def report(self) -> Dict[str, float]:
        return {f"{name}_ms_per_call": self.mean_ms(name) for name in sorted(self.totals)}

    def __str__(self) -> str:
        return "  ".join(f"{k}={v:.2f}ms" for k, v in self.report().items())


@contextmanager
def timed(label: str, logger=None) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        message = f"{label}: {elapsed * 1000:.2f} ms"
        (logger.info if logger else print)(message)
