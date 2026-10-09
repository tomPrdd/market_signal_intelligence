"""The extraction loop: chunk, call the model, validate, dedup, persist."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from tqdm import tqdm

from .backends.base import LLMBackend
from .chunking import Chunk, split_document
from .exceptions import LLMBackendError
from .io import append_to_pool, existing_source_ids
from .prompts import (
    FINANCE_PROMPT_VERSION,
    FINANCE_SYSTEM_PROMPT_V1_1,
    FINANCE_USER_PROMPT_TEMPLATE,
)
from .schema import PTC
from .validation import filter_valid_ptcs

logger = logging.getLogger(__name__)


class PTCExtractor:
    """Extract PTCs from documents using an :class:`LLMBackend`.

    The prompt is injectable. With no prompt arguments you get the finance
    prompt this library was built around; pass ``system_prompt``,
    ``user_prompt_template`` and ``extraction_version`` to target another
    domain.

    The user prompt template is formatted with ``source_date``, ``source_type``
    and ``chunk_text``. A template using only a subset of those is fine; one
    referencing anything else raises ``KeyError`` at extraction time.
    """

    def __init__(
        self,
        backend: LLMBackend,
        max_workers: int = 8,
        max_chunk_chars: int = 8000,
        overlap_chars: int = 400,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        extraction_version: str | None = None,
    ):
        self._backend = backend
        self._max_workers = max_workers
        self._max_chunk_chars = max_chunk_chars
        self._overlap_chars = overlap_chars
        self._system_prompt = system_prompt or FINANCE_SYSTEM_PROMPT_V1_1
        self._user_prompt_template = user_prompt_template or FINANCE_USER_PROMPT_TEMPLATE
        self._extraction_version = extraction_version or FINANCE_PROMPT_VERSION

    @property
    def extraction_version(self) -> str:
        """The version label stamped onto every PTC this extractor produces."""
        return self._extraction_version

    def _locate_span(self, raw_text: str, chunk: Chunk) -> tuple[int, int] | None:
        """Find raw_text's character offsets in the original document.

        Computed, never asked for: models are unreliable at character offsets,
        and the chunker already guarantees ``document[chunk.start:chunk.end] ==
        chunk.text``, so locating the quote inside the chunk yields exact
        document-level offsets for free.

        Returns None when the quote is not verbatim in the chunk — a meaningful
        signal that the model paraphrased rather than quoted.
        """
        if not raw_text:
            return None
        idx = chunk.text.find(raw_text)
        if idx == -1:
            return None
        start = chunk.start + idx
        return (start, start + len(raw_text))

    def _extract_from_chunk(
        self,
        chunk: Chunk,
        source_id: str,
        source_type: str,
        source_date: date,
    ) -> tuple[list[PTC], int]:
        """Extract from one chunk. Returns (valid PTCs, number dropped)."""
        user_prompt = self._user_prompt_template.format(
            source_date=source_date.isoformat(),
            source_type=str(source_type),
            chunk_text=chunk.text,
        )
        try:
            response_text = self._backend.complete(self._system_prompt, user_prompt)
        except LLMBackendError:
            logger.exception("LLM backend failed for source_id=%s", source_id)
            raise

        # Strip code fences if the model emits any despite instructions.
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            raw_list = json.loads(cleaned)
            if not isinstance(raw_list, list):
                logger.warning("Non-list JSON from LLM, source_id=%s: %r", source_id, raw_list)
                return [], 0
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON for source_id=%s: %r", source_id, cleaned)
            return [], 0

        # Offsets are derived from the text we sent, not requested from the model.
        for raw in raw_list:
            if isinstance(raw, dict):
                raw["span"] = self._locate_span(raw.get("raw_text") or "", chunk)

        valid = filter_valid_ptcs(
            raw_list,
            source_id=source_id,
            source_type=source_type,
            source_date=source_date,
            extracted_by=self._backend.model_id,
            extraction_version=self._extraction_version,
        )
        return valid, len(raw_list) - len(valid)

    def extract_from_text(
        self,
        text: str,
        source_id: str,
        source_type: str,
        source_date: date,
    ) -> list[PTC]:
        """Extract every PTC from one document's text."""
        ptcs, _ = self.extract_from_text_with_stats(text, source_id, source_type, source_date)
        return ptcs

    def extract_from_text_with_stats(
        self,
        text: str,
        source_id: str,
        source_type: str,
        source_date: date,
    ) -> tuple[list[PTC], int]:
        """Like :meth:`extract_from_text`, but also returns how many were dropped.

        Malformed model output is dropped, never repaired — this is how you find
        out how much was dropped.
        """
        chunks = split_document(text, self._max_chunk_chars, self._overlap_chars)
        if not chunks:
            return [], 0

        results: list[PTC] = []
        dropped = 0
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [
                pool.submit(self._extract_from_chunk, c, source_id, source_type, source_date)
                for c in chunks
            ]
            for fut in as_completed(futures):
                chunk_ptcs, chunk_dropped = fut.result()
                results.extend(chunk_ptcs)
                dropped += chunk_dropped

        # Dedup intra-document by content_hash: overlapping chunks mean the same
        # claim is often seen twice. When duplicates disagree on span — the quote
        # sits whole in one chunk and straddles the boundary in another — keep the
        # copy that carries offsets.
        by_hash: dict[str, PTC] = {}
        for ptc in results:
            h = ptc.content_hash()
            kept = by_hash.get(h)
            if kept is None or (kept.span is None and ptc.span is not None):
                by_hash[h] = ptc
        return list(by_hash.values()), dropped

    def extract_from_documents(
        self,
        documents: list[dict],
        output_path: Path,
        resume: bool = True,
    ) -> dict:
        """Extract from many documents, appending to a JSONL pool as it goes.

        Each document dict: ``{source_id, text, source_type, source_date}``.

        With ``resume=True`` any source_id already in the pool is skipped, so an
        interrupted run restarts without re-paying for documents it already
        processed.
        """
        already_done: set[str] = existing_source_ids(output_path) if resume else set()

        stats = {"processed": 0, "skipped": 0, "extracted": 0, "dropped": 0, "errors": 0}

        for doc in tqdm(documents, desc="Extracting PTCs"):
            sid = doc["source_id"]
            if sid in already_done:
                stats["skipped"] += 1
                continue
            try:
                ptcs, dropped = self.extract_from_text_with_stats(
                    text=doc["text"],
                    source_id=sid,
                    source_type=doc["source_type"],
                    source_date=doc["source_date"],
                )
                append_to_pool(output_path, ptcs)
                stats["processed"] += 1
                stats["extracted"] += len(ptcs)
                stats["dropped"] += dropped
            except LLMBackendError:
                stats["errors"] += 1
                logger.exception("Failed to process source_id=%s", sid)

        return stats
