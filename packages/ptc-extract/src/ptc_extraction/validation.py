"""Drop-on-error validation.

A raw dict from the model either becomes a valid PTC or it is discarded and
logged. Nothing is coerced into shape: a corpus of hundreds of documents always
produces some malformed output, and silently repairing it is how bad claims end
up in results.
"""

import logging
from datetime import date, datetime
from typing import Any

from .schema import PTC, Direction, Polarity

logger = logging.getLogger(__name__)


def _parse_event_date(value: Any) -> date | None:
    if value is None or value == "" or value == "null":
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def validate_raw_ptc_dict(
    raw: dict,
    source_id: str,
    source_type: str,
    source_date: date,
    extracted_by: str,
    extraction_version: str,
) -> PTC | None:
    """Construct a PTC from a raw model dict. Return None on validation failure.

    ``source_type`` is any non-empty string — the corpus label is yours to
    choose. See :class:`~ptc_extraction.schema.SourceType` for the conventional
    two-corpus values.
    """
    try:
        polarity_raw = raw.get("polarity")
        polarity = Polarity(int(polarity_raw)) if polarity_raw is not None else None
        direction = Direction(raw["direction"]) if "direction" in raw else None
        if direction is None or polarity is None:
            return None
        return PTC(
            mechanism=raw["mechanism"],
            raw_text=raw["raw_text"],
            direction=direction,
            polarity=polarity,
            source_date=source_date,
            event_date=_parse_event_date(raw.get("event_date")),
            source_type=source_type,
            source_id=source_id,
            span=raw.get("span"),
            extracted_by=extracted_by,
            extraction_version=extraction_version,
            confidence=raw.get("confidence"),
        )
    except (KeyError, ValueError, TypeError) as e:
        logger.warning("Dropped invalid PTC dict (source=%s): %s — raw=%r", source_id, e, raw)
        return None


def filter_valid_ptcs(
    raw_list: list[dict],
    source_id: str,
    source_type: str,
    source_date: date,
    extracted_by: str,
    extraction_version: str,
) -> list[PTC]:
    """Convert raw dicts to validated PTCs, dropping malformed ones.

    The caller gets back only what survived; compare against ``len(raw_list)``
    to count what was dropped.
    """
    result = []
    for raw in raw_list:
        ptc = validate_raw_ptc_dict(
            raw, source_id, source_type, source_date, extracted_by, extraction_version
        )
        if ptc is not None:
            result.append(ptc)
    return result
