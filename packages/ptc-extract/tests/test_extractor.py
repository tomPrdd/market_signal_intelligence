import json
from datetime import date
from unittest.mock import MagicMock

from ptc_extraction.exceptions import LLMBackendError
from ptc_extraction.extractor import PTCExtractor
from ptc_extraction.schema import Direction, Polarity, SourceType

VALID_LLM_RESPONSE = json.dumps(
    [
        {
            "mechanism": "Rising raw material costs compress operating margins over subsequent quarters.",
            "raw_text": "Raw material costs surged in Q3 2024, pressuring our operating margins.",
            "direction": "precursor",
            "polarity": -1,
            "confidence": 0.9,
        }
    ]
)

DUPLICATE_LLM_RESPONSE = json.dumps(
    [
        {
            "mechanism": "Rising raw material costs compress operating margins over subsequent quarters.",
            "raw_text": "Raw material costs surged in Q3 2024, pressuring our operating margins.",
            "direction": "precursor",
            "polarity": -1,
            "confidence": 0.9,
        },
        {
            "mechanism": "Rising raw material costs compress operating margins over subsequent quarters.",
            "raw_text": "Raw material costs surged in Q3 2024, pressuring our operating margins.",
            "direction": "precursor",
            "polarity": -1,
            "confidence": 0.9,
        },
    ]
)


def _make_backend(return_value: str = VALID_LLM_RESPONSE) -> MagicMock:
    backend = MagicMock()
    backend.model_id = "eu.anthropic.claude-sonnet-4-6"
    backend.complete.return_value = return_value
    return backend


def _make_extractor(backend=None, max_chunk_chars: int = 8000) -> PTCExtractor:
    if backend is None:
        backend = _make_backend()
    return PTCExtractor(backend=backend, max_chunk_chars=max_chunk_chars, max_workers=1)


def test_single_chunk_valid_response_returns_ptcs():
    backend = _make_backend(VALID_LLM_RESPONSE)
    extractor = _make_extractor(backend)
    ptcs = extractor.extract_from_text(
        text="Raw material costs surged in Q3 2024, pressuring our operating margins.",
        source_id="press-001",
        source_type=SourceType.PRESS,
        source_date=date(2024, 10, 15),
    )
    assert len(ptcs) == 1
    assert ptcs[0].direction == Direction.PRECURSOR
    assert ptcs[0].polarity == Polarity.NEGATIVE
    assert ptcs[0].source_id == "press-001"


def test_malformed_json_returns_empty_list(caplog):
    backend = _make_backend("this is not json at all {{")
    extractor = _make_extractor(backend)
    ptcs = extractor.extract_from_text(
        text="Some text.",
        source_id="press-002",
        source_type=SourceType.PRESS,
        source_date=date(2024, 10, 15),
    )
    assert ptcs == []


def test_backend_error_in_extract_from_documents_increments_errors(tmp_path):
    backend = _make_backend()
    backend.complete.side_effect = LLMBackendError("Connection timeout")
    extractor = _make_extractor(backend)

    docs = [
        {
            "source_id": "press-error-001",
            "text": "Some news article text here.",
            "source_type": "press",
            "source_date": date(2024, 10, 15),
        }
    ]
    output = tmp_path / "pool.jsonl"
    stats = extractor.extract_from_documents(docs, output)
    assert stats["errors"] == 1
    assert stats["processed"] == 0


def test_resume_skips_existing_source_ids(tmp_path):
    backend = _make_backend(VALID_LLM_RESPONSE)
    extractor = _make_extractor(backend)
    output = tmp_path / "pool.jsonl"

    docs = [
        {
            "source_id": "press-001",
            "text": "Raw material costs surged in Q3 2024.",
            "source_type": "press",
            "source_date": date(2024, 10, 15),
        },
        {
            "source_id": "press-002",
            "text": "Revenue growth exceeded expectations this quarter.",
            "source_type": "press",
            "source_date": date(2024, 10, 16),
        },
    ]
    # Process press-001 first
    stats1 = extractor.extract_from_documents([docs[0]], output, resume=True)
    assert stats1["processed"] == 1

    # Now run both; press-001 should be skipped
    stats2 = extractor.extract_from_documents(docs, output, resume=True)
    assert stats2["skipped"] == 1
    assert stats2["processed"] == 1


def test_intra_document_dedup_keeps_one_unique(tmp_path):
    """Two chunks returning same content_hash → only one PTC kept."""
    backend = _make_backend(DUPLICATE_LLM_RESPONSE)
    # Force two chunks by using tiny max_chunk_chars — text will split
    long_text = (
        ("Raw material costs surged in Q3 2024. " * 50)
        + "\n\n"
        + ("Operating margins declined sharply. " * 50)
    )
    extractor = PTCExtractor(backend=backend, max_chunk_chars=500, max_workers=1)
    ptcs = extractor.extract_from_text(
        text=long_text,
        source_id="press-dedup-001",
        source_type=SourceType.PRESS,
        source_date=date(2024, 10, 15),
    )
    # Both chunks return the same mechanism — dedup should collapse them
    hashes = [p.content_hash() for p in ptcs]
    assert len(hashes) == len(set(hashes)), "Duplicate content_hashes found after dedup"


def test_non_list_json_returns_empty_list():
    backend = _make_backend('{"mechanism": "single object not a list"}')
    extractor = _make_extractor(backend)
    ptcs = extractor.extract_from_text(
        text="Some text.",
        source_id="press-003",
        source_type=SourceType.PRESS,
        source_date=date(2024, 10, 15),
    )
    assert ptcs == []


def test_resume_false_does_not_skip(tmp_path):
    backend = _make_backend(VALID_LLM_RESPONSE)
    extractor = _make_extractor(backend)
    output = tmp_path / "pool.jsonl"

    doc = {
        "source_id": "press-001",
        "text": "Raw material costs surged.",
        "source_type": "press",
        "source_date": date(2024, 10, 15),
    }
    extractor.extract_from_documents([doc], output, resume=True)
    stats = extractor.extract_from_documents([doc], output, resume=False)
    assert stats["skipped"] == 0
    assert stats["processed"] == 1


def test_event_date_quarter_iso_passthrough():
    """LLM returns event_date="2024-07-01" for a Q3 mention; pipeline preserves it as-is.

    v1.1 prompt instructs the LLM to map "Q3 2024" → "2024-07-01".
    The Python layer does not perform this mapping — it just validates the ISO string.
    This test documents the expected end-to-end behaviour when the LLM complies.
    """
    q3_response = json.dumps(
        [
            {
                "mechanism": "Red Sea shipping disruptions added €45M in logistics costs and management does not expect resolution before mid-2025.",
                "raw_text": "Red Sea shipping disruptions began in Q3 2024, adding €45M in costs.",
                "direction": "precursor",
                "polarity": -1,
                "event_date": "2024-07-01",
                "confidence": 0.94,
            }
        ]
    )
    backend = _make_backend(q3_response)
    extractor = _make_extractor(backend)
    ptcs = extractor.extract_from_text(
        text="Red Sea shipping disruptions began in Q3 2024, adding €45M in costs.",
        source_id="press-redsea-001",
        source_type=SourceType.PRESS,
        source_date=date(2024, 11, 14),
    )
    assert len(ptcs) == 1
    from datetime import date as _date

    assert ptcs[0].event_date == _date(2024, 7, 1)
    assert ptcs[0].direction == Direction.PRECURSOR
    assert ptcs[0].extraction_version == "v1.1"


def test_atomicity_two_ptcs_from_single_chunk():
    """LLM returns two PTCs (consequence + precursor) for one passage; both are preserved.

    v1.1 prompt instructs the LLM to split temporally distinct claims. This test verifies
    the pipeline correctly passes both through without collapsing them.
    """
    split_response = json.dumps(
        [
            {
                "mechanism": "Euro strengthening compressed Q3 reported revenue by €120M — already fully realized in closed quarter.",
                "raw_text": "The stronger euro compressed Q3 revenue by €120M.",
                "direction": "consequence",
                "polarity": -1,
                "event_date": "2024-07-01",
                "confidence": 0.95,
            },
            {
                "mechanism": "Forex headwinds from euro strength are expected to persist into Q4, creating continued revenue drag.",
                "raw_text": "Forex headwinds are expected to persist into Q4.",
                "direction": "precursor",
                "polarity": -1,
                "event_date": None,
                "confidence": 0.88,
            },
        ]
    )
    backend = _make_backend(split_response)
    extractor = _make_extractor(backend)
    ptcs = extractor.extract_from_text(
        text="The stronger euro compressed Q3 revenue by €120M; forex headwinds are expected to persist into Q4.",
        source_id="press-forex-001",
        source_type=SourceType.PRESS,
        source_date=date(2024, 11, 14),
    )
    assert len(ptcs) == 2
    directions = {p.direction for p in ptcs}
    assert Direction.PRECURSOR in directions
    assert Direction.CONSEQUENCE in directions
    # Both belong to same source and use current prompt version
    assert all(p.extraction_version == "v1.1" for p in ptcs)
