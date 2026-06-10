"""
logger.py — RAG system logger (Loguru-based)

Usage:
    from logger import log

    log.info("Starting ingestion")
    log.info("Search done", query="Article 5", hits=5)
    log.error("Azure failed", exc=e, query="Article 5")

    with log.timer("embedding batch"):
        embeddings = embed(chunks)
"""

import os
import sys
from contextlib import contextmanager
from time import perf_counter

from loguru import logger as _logger
from gtgh_team3_compliance_assistant.config import LOGS_FILE


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_logger.remove()  # Remove default handler

_logger.add(
    sys.stderr,
    level=os.getenv("RAG_LOG_LEVEL", "INFO"),
    format="<green>{time:YYYY-MM-DDTHH:mm:ss}Z</green>  <level>{level: <8}</level>  <cyan>[{name}]</cyan>  {message}  {extra}",
    colorize=True,
)

if log_file := LOGS_FILE:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    _logger.add(
        log_file,
        level=os.getenv("RAG_LOG_LEVEL", "INFO"),
        serialize=True,          # writes newline-delimited JSON
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
    )

class RagLogger:
    """Thin wrapper around loguru with RAG-specific helpers."""

    def __init__(self, name: str = "rag") -> None:
        self._log = _logger.bind(logger=name)

    # Standard levels
    def debug(self, msg: str, **kw) -> None:
        self._log.opt(depth=1).debug(msg, **kw)

    def info(self, msg: str, **kw) -> None:
        self._log.opt(depth=1).info(msg, **kw)

    def warning(self, msg: str, **kw) -> None:
        self._log.opt(depth=1).warning(msg, **kw)

    def error(self, msg: str, exc: BaseException | None = None, **kw) -> None:
        self._log.opt(depth=1, exception=exc).error(msg, **kw)

    def critical(self, msg: str, exc: BaseException | None = None, **kw) -> None:
        self._log.opt(depth=1, exception=exc).critical(msg, **kw)

    # Domain helpers
    def log_ingestion(self, filename: str, chunks: int, skipped: int = 0, duration_s: float = 0.0) -> None:
        self._log.opt(depth=1).info(
            f"Ingestion complete: {filename}",
            event="ingestion", chunks=chunks, skipped=skipped, duration_s=round(duration_s, 3),
        )

    def log_embedding(self, batch_size: int, duration_s: float, model: str | None = None) -> None:
        self._log.opt(depth=1).info(
            f"Embedded {batch_size} chunks",
            event="embedding", batch_size=batch_size, model=model, duration_s=round(duration_s, 3),
        )

    def log_indexing(self, documents_uploaded: int, index_name: str, duration_s: float) -> None:
        self._log.opt(depth=1).info(
            f"Indexed {documents_uploaded} docs into '{index_name}'",
            event="indexing", index_name=index_name, docs=documents_uploaded, duration_s=round(duration_s, 3),
        )

    def log_search(self, query: str, hits: int, scores: list[float] | None = None, duration_s: float | None = None) -> None:
        self._log.opt(depth=1).info(
            f"Search returned {hits} hit(s)",
            event="search", query=query, hits=hits,
            top_score=max(scores) if scores else None,
            duration_s=round(duration_s, 3) if duration_s else None,
        )

    def log_llm_call(self, question: str, prompt_tokens: int, completion_tokens: int, duration_s: float, rule_applied: str | None = None) -> None:
        self._log.opt(depth=1).info(
            f"LLM call complete ({prompt_tokens + completion_tokens} tokens, {duration_s:.1f}s)",
            event="llm_call", question=question[:120], prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, rule=rule_applied, duration_s=round(duration_s, 3),
        )

    def log_llm_no_context(self, question: str) -> None:
        self._log.opt(depth=1).warning(
            "Rule E triggered — no usable context",
            event="llm_no_context", question=question[:120],
        )

    # Timer context manager
    @contextmanager
    def timer(self, label: str, **kw):
        start = perf_counter()
        self._log.opt(depth=2).debug(f"Starting: {label}")
        try:
            yield
            self._log.opt(depth=2).info(f"Completed: {label}", event="timer", label=label, duration_s=round(perf_counter() - start, 3), **kw)
        except Exception as exc:
            self._log.opt(depth=2, exception=exc).error(f"Failed: {label}", event="timer_error", label=label, duration_s=round(perf_counter() - start, 3), **kw)
            raise

log = RagLogger(name="rag")