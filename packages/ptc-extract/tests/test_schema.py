from datetime import date

import pytest
from pydantic import ValidationError

from ptc_extraction.schema import PTC, Direction, Polarity, SourceType


def _valid_ptc(**overrides) -> PTC:
    base = dict(
        mechanism="Rising raw material costs compress operating margins significantly.",
        raw_text="Raw material costs surged in Q3 2024.",
        direction=Direction.PRECURSOR,
        polarity=Polarity.NEGATIVE,
        source_date=date(2024, 10, 15),
        source_type=SourceType.PRESS,
        source_id="press-001",
        extracted_by="eu.anthropic.claude-sonnet-4-6",
        extraction_version="v1.1",
    )
    base.update(overrides)
    return PTC(**base)


def test_valid_ptc_construction():
    ptc = _valid_ptc()
    assert ptc.direction == Direction.PRECURSOR
    assert ptc.polarity == Polarity.NEGATIVE
    assert ptc.source_id == "press-001"


def test_event_date_after_source_date_raises():
    with pytest.raises(ValidationError):
        _valid_ptc(
            source_date=date(2024, 10, 15),
            event_date=date(2024, 10, 16),
        )


def test_mechanism_too_short_raises():
    with pytest.raises(ValidationError):
        _valid_ptc(mechanism="Short")


def test_invalid_polarity_raises():
    with pytest.raises(ValidationError):
        _valid_ptc(polarity=0)


def test_content_hash_is_deterministic():
    ptc = _valid_ptc()
    assert ptc.content_hash() == ptc.content_hash()


def test_content_hash_differs_on_polarity_change():
    ptc_neg = _valid_ptc(polarity=Polarity.NEGATIVE)
    ptc_pos = _valid_ptc(polarity=Polarity.POSITIVE)
    assert ptc_neg.content_hash() != ptc_pos.content_hash()


def test_event_date_equal_to_source_date_allowed():
    ptc = _valid_ptc(
        source_date=date(2024, 10, 15),
        event_date=date(2024, 10, 15),
    )
    assert ptc.event_date == date(2024, 10, 15)


def test_confidence_out_of_range_raises():
    with pytest.raises(ValidationError):
        _valid_ptc(confidence=1.5)
