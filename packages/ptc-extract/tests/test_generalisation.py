"""Domain-agnostic behaviour: arbitrary source_type, injectable prompts, spans."""

import json
from datetime import date
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from ptc_extraction import PTC, PTCExtractor, SourceType
from ptc_extraction.prompts import (
    FINANCE_PROMPT_VERSION,
    FINANCE_SYSTEM_PROMPT_V1_1,
    FINANCE_USER_PROMPT_TEMPLATE,
    PROMPT_VERSION,
    SYSTEM_PROMPT_V1,
    USER_PROMPT_TEMPLATE_V1,
)

QUOTE = "Raw material costs surged in Q3 2024, pressuring our operating margins."

RESPONSE = json.dumps(
    [
        {
            "mechanism": "Rising raw material costs compress operating margins.",
            "raw_text": QUOTE,
            "direction": "precursor",
            "polarity": -1,
            "confidence": 0.9,
        }
    ]
)


def _backend(return_value: str = RESPONSE) -> MagicMock:
    backend = MagicMock()
    backend.model_id = "test-model"
    backend.complete.return_value = return_value
    return backend


def _ptc(**overrides) -> dict:
    base = {
        "mechanism": "Rising raw material costs compress operating margins.",
        "raw_text": QUOTE,
        "direction": "precursor",
        "polarity": -1,
        "source_date": date(2024, 11, 14),
        "source_type": "management",
        "source_id": "DOC-1",
        "extracted_by": "test-model",
        "extraction_version": "v1.1",
    }
    return {**base, **overrides}


# --- source_type generalisation -------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["management", "press", "abstract", "statute", "ruling", "clinical-trial", "Ministère"],
)
def test_source_type_accepts_any_non_empty_string(value):
    assert PTC(**_ptc(source_type=value)).source_type == value


def test_source_type_rejects_empty_string():
    with pytest.raises(ValidationError):
        PTC(**_ptc(source_type=""))


def test_source_type_enum_members_still_work_and_serialise_identically():
    from_enum = PTC(**_ptc(source_type=SourceType.MANAGEMENT))
    from_str = PTC(**_ptc(source_type="management"))
    assert from_enum.source_type == from_str.source_type == "management"
    assert from_enum.model_dump_json() == from_str.model_dump_json()
    assert '"source_type":"management"' in from_enum.model_dump_json()


def test_source_type_enum_constants_unchanged():
    assert SourceType.MANAGEMENT == "management"
    assert SourceType.PRESS == "press"


def test_content_hash_unaffected_by_source_type_representation():
    """content_hash is a stable contract — it must not shift under this change."""
    a = PTC(**_ptc(source_type=SourceType.PRESS))
    b = PTC(**_ptc(source_type="press"))
    assert a.content_hash() == b.content_hash()
    # Pinned value: any change here breaks dedup against every existing pool.
    assert a.content_hash() == "4d8847b0cc150171"


def test_serialisation_is_byte_stable():
    """The JSONL format is a contract: a pool line must round-trip unchanged."""
    line = (
        '{"mechanism":"Rising raw material costs compress operating margins.",'
        f'"raw_text":"{QUOTE}","direction":"precursor","polarity":-1,'
        '"source_date":"2024-11-14","event_date":null,"source_type":"management",'
        '"source_id":"DOC-1","span":null,"extracted_by":"test-model",'
        '"extraction_version":"v1.1","confidence":null}'
    )
    assert PTC.model_validate_json(line).model_dump_json() == line


def test_dedup_prefers_the_copy_that_carries_a_span():
    """Overlapping chunks can locate the same claim in one and not the other."""
    with_span = PTC(**_ptc(span=(10, 40)))
    without = PTC(**_ptc())
    assert with_span.content_hash() == without.content_hash()


def test_extractor_passes_arbitrary_source_type_through():
    backend = _backend()
    extractor = PTCExtractor(backend=backend, max_workers=1)
    ptcs = extractor.extract_from_text(
        text=QUOTE, source_id="DOC-1", source_type="abstract", source_date=date(2024, 11, 14)
    )
    assert ptcs[0].source_type == "abstract"
    assert "Document type: abstract" in backend.complete.call_args[0][1]


# --- injectable prompts ---------------------------------------------------------


def test_default_prompt_is_the_finance_one():
    extractor = PTCExtractor(backend=_backend(), max_workers=1)
    assert extractor.extraction_version == FINANCE_PROMPT_VERSION == "v1.1"


def test_legacy_prompt_aliases_still_resolve():
    assert SYSTEM_PROMPT_V1 is FINANCE_SYSTEM_PROMPT_V1_1
    assert USER_PROMPT_TEMPLATE_V1 is FINANCE_USER_PROMPT_TEMPLATE
    assert PROMPT_VERSION == FINANCE_PROMPT_VERSION


def test_custom_prompt_and_version_are_used():
    backend = _backend()
    extractor = PTCExtractor(
        backend=backend,
        max_workers=1,
        system_prompt="You extract causal claims from clinical abstracts.",
        user_prompt_template="Published {source_date}. Corpus {source_type}.\n{chunk_text}",
        extraction_version="clinical-v2",
    )
    ptcs = extractor.extract_from_text(
        text=QUOTE, source_id="DOC-1", source_type="abstract", source_date=date(2024, 11, 14)
    )
    system, user = backend.complete.call_args[0]
    assert system == "You extract causal claims from clinical abstracts."
    assert user.startswith("Published 2024-11-14. Corpus abstract.")
    assert "financial analyst" not in system
    assert ptcs[0].extraction_version == "clinical-v2"


def test_custom_template_may_ignore_some_placeholders():
    backend = _backend()
    extractor = PTCExtractor(
        backend=backend, max_workers=1, user_prompt_template="Text only:\n{chunk_text}"
    )
    extractor.extract_from_text(
        text=QUOTE, source_id="DOC-1", source_type="abstract", source_date=date(2024, 11, 14)
    )
    assert backend.complete.call_args[0][1] == f"Text only:\n{QUOTE}"


# --- span ------------------------------------------------------------------------


def test_span_is_computed_and_points_at_the_quote():
    prefix = "Some preamble that pushes the quote along. "
    document = prefix + QUOTE + " And a trailing sentence."
    extractor = PTCExtractor(backend=_backend(), max_workers=1)
    ptcs = extractor.extract_from_text(
        text=document, source_id="DOC-1", source_type="press", source_date=date(2024, 11, 14)
    )
    start, end = ptcs[0].span
    assert document[start:end] == QUOTE
    assert start == len(prefix)


def test_span_is_none_when_the_model_paraphrases():
    paraphrased = json.dumps(
        [
            {
                "mechanism": "Rising raw material costs compress operating margins.",
                "raw_text": "A sentence that never appears in the source document at all.",
                "direction": "precursor",
                "polarity": -1,
            }
        ]
    )
    extractor = PTCExtractor(backend=_backend(paraphrased), max_workers=1)
    ptcs = extractor.extract_from_text(
        text=QUOTE, source_id="DOC-1", source_type="press", source_date=date(2024, 11, 14)
    )
    assert ptcs[0].span is None


def test_span_offsets_are_document_relative_across_chunks():
    """The quote sits in a later chunk; offsets must still index the whole document."""
    filler = "\n\n".join(f"Paragraph number {i} of unrelated filler text." for i in range(40))
    document = filler + "\n\n" + QUOTE
    extractor = PTCExtractor(backend=_backend(), max_workers=1, max_chunk_chars=300)
    ptcs = extractor.extract_from_text(
        text=document, source_id="DOC-1", source_type="press", source_date=date(2024, 11, 14)
    )
    spans = {p.span for p in ptcs if p.span}
    assert spans, "expected at least one located span"
    for start, end in spans:
        assert document[start:end] == QUOTE
        assert start > 500, "span must be document-relative, not chunk-relative"


def test_model_supplied_span_is_ignored_in_favour_of_the_computed_one():
    lying = json.dumps(
        [
            {
                "mechanism": "Rising raw material costs compress operating margins.",
                "raw_text": QUOTE,
                "direction": "precursor",
                "polarity": -1,
                "span": [9999, 10042],
            }
        ]
    )
    extractor = PTCExtractor(backend=_backend(lying), max_workers=1)
    ptcs = extractor.extract_from_text(
        text=QUOTE, source_id="DOC-1", source_type="press", source_date=date(2024, 11, 14)
    )
    assert ptcs[0].span == (0, len(QUOTE))


@pytest.mark.parametrize("bad", [(-1, 5), (10, 3)])
def test_schema_rejects_nonsensical_spans(bad):
    with pytest.raises(ValidationError):
        PTC(**_ptc(span=bad))


def test_span_round_trips_through_json():
    ptc = PTC(**_ptc(span=(12, 84)))
    assert PTC.model_validate_json(ptc.model_dump_json()).span == (12, 84)


# --- drop-and-count ---------------------------------------------------------------


def test_dropped_claims_are_counted_not_repaired():
    mixed = json.dumps(
        [
            {
                "mechanism": "Rising raw material costs compress operating margins.",
                "raw_text": QUOTE,
                "direction": "precursor",
                "polarity": -1,
            },
            {"mechanism": "too short", "raw_text": QUOTE, "direction": "precursor", "polarity": -1},
            {
                "mechanism": "A mechanism with an unusable polarity value entirely.",
                "raw_text": QUOTE,
                "direction": "precursor",
                "polarity": "upwards",
            },
        ]
    )
    extractor = PTCExtractor(backend=_backend(mixed), max_workers=1)
    ptcs, dropped = extractor.extract_from_text_with_stats(
        text=QUOTE, source_id="DOC-1", source_type="press", source_date=date(2024, 11, 14)
    )
    assert len(ptcs) == 1
    assert dropped == 2


def test_extract_from_documents_reports_dropped(tmp_path):
    partly_bad = json.dumps(
        [
            {
                "mechanism": "Rising raw material costs compress operating margins.",
                "raw_text": QUOTE,
                "direction": "precursor",
                "polarity": -1,
            },
            {"mechanism": "nope", "raw_text": QUOTE, "direction": "precursor", "polarity": -1},
        ]
    )
    extractor = PTCExtractor(backend=_backend(partly_bad), max_workers=1)
    stats = extractor.extract_from_documents(
        documents=[
            {
                "source_id": "DOC-1",
                "text": QUOTE,
                "source_type": "abstract",
                "source_date": date(2024, 11, 14),
            }
        ],
        output_path=tmp_path / "pool.jsonl",
    )
    assert stats == {
        "processed": 1,
        "skipped": 0,
        "extracted": 1,
        "dropped": 1,
        "errors": 0,
    }
