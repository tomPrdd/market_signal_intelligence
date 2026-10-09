from datetime import date

from ptc_extraction.schema import Direction, Polarity, SourceType
from ptc_extraction.validation import validate_raw_ptc_dict


def _common_kwargs():
    return dict(
        source_id="doc-001",
        source_type=SourceType.PRESS,
        source_date=date(2024, 10, 15),
        extracted_by="eu.anthropic.claude-sonnet-4-6",
        extraction_version="v1.1",
    )


def _valid_raw():
    return {
        "mechanism": "Rising input costs erode operating margins over the following quarters.",
        "raw_text": "Input costs rose sharply in Q3.",
        "direction": "precursor",
        "polarity": -1,
        "confidence": 0.85,
    }


def test_valid_dict_produces_ptc():
    ptc = validate_raw_ptc_dict(_valid_raw(), **_common_kwargs())
    assert ptc is not None
    assert ptc.direction == Direction.PRECURSOR
    assert ptc.polarity == Polarity.NEGATIVE


def test_missing_direction_returns_none():
    raw = _valid_raw()
    del raw["direction"]
    result = validate_raw_ptc_dict(raw, **_common_kwargs())
    assert result is None


def test_event_date_after_source_date_returns_none():
    raw = _valid_raw()
    raw["event_date"] = "2025-01-01"
    kwargs = _common_kwargs()
    kwargs["source_date"] = date(2024, 10, 15)
    result = validate_raw_ptc_dict(raw, **kwargs)
    assert result is None


def test_polarity_as_string_numeric_coerced():
    raw = _valid_raw()
    raw["polarity"] = "1"
    ptc = validate_raw_ptc_dict(raw, **_common_kwargs())
    assert ptc is not None
    assert ptc.polarity == Polarity.POSITIVE


def test_event_date_as_null_string_resolves_to_none():
    raw = _valid_raw()
    raw["event_date"] = "null"
    ptc = validate_raw_ptc_dict(raw, **_common_kwargs())
    assert ptc is not None
    assert ptc.event_date is None


def test_event_date_as_empty_string_resolves_to_none():
    raw = _valid_raw()
    raw["event_date"] = ""
    ptc = validate_raw_ptc_dict(raw, **_common_kwargs())
    assert ptc is not None
    assert ptc.event_date is None


def test_event_date_missing_resolves_to_none():
    raw = _valid_raw()
    ptc = validate_raw_ptc_dict(raw, **_common_kwargs())
    assert ptc is not None
    assert ptc.event_date is None


def test_missing_polarity_returns_none():
    raw = _valid_raw()
    del raw["polarity"]
    result = validate_raw_ptc_dict(raw, **_common_kwargs())
    assert result is None


def test_invalid_direction_value_returns_none():
    raw = _valid_raw()
    raw["direction"] = "sideways"
    result = validate_raw_ptc_dict(raw, **_common_kwargs())
    assert result is None
