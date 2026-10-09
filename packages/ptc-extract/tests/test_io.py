from datetime import date

from ptc_extraction.io import append_to_pool, dedup_pool, existing_source_ids, load_pool
from ptc_extraction.schema import PTC, Direction, Polarity, SourceType


def _make_ptc(source_id: str = "src-001", mechanism_suffix: str = "") -> PTC:
    return PTC(
        mechanism=f"Rising input costs erode operating margins significantly{mechanism_suffix}.",
        raw_text="Input costs rose sharply in Q3.",
        direction=Direction.PRECURSOR,
        polarity=Polarity.NEGATIVE,
        source_date=date(2024, 10, 15),
        source_type=SourceType.PRESS,
        source_id=source_id,
        extracted_by="eu.anthropic.claude-sonnet-4-6",
        extraction_version="v1.1",
        confidence=0.9,
    )


def test_load_pool_missing_file_returns_empty(tmp_path):
    result = load_pool(tmp_path / "nonexistent.jsonl")
    assert result == []


def test_roundtrip_append_then_load(tmp_path):
    pool_path = tmp_path / "pool.jsonl"
    ptc = _make_ptc()
    append_to_pool(pool_path, [ptc])
    loaded = load_pool(pool_path)
    assert len(loaded) == 1
    assert loaded[0].mechanism == ptc.mechanism
    assert loaded[0].source_id == ptc.source_id
    assert loaded[0].polarity == ptc.polarity


def test_append_creates_file_if_missing(tmp_path):
    pool_path = tmp_path / "sub" / "pool.jsonl"
    ptc = _make_ptc()
    count = append_to_pool(pool_path, [ptc])
    assert count == 1
    assert pool_path.exists()


def test_existing_source_ids_returns_expected_set(tmp_path):
    pool_path = tmp_path / "pool.jsonl"
    ptcs = [_make_ptc("src-001"), _make_ptc("src-002", "_b")]
    append_to_pool(pool_path, ptcs)
    ids = existing_source_ids(pool_path)
    assert ids == {"src-001", "src-002"}


def test_existing_source_ids_missing_file_returns_empty(tmp_path):
    ids = existing_source_ids(tmp_path / "none.jsonl")
    assert ids == set()


def test_dedup_pool_removes_duplicates(tmp_path):
    pool_path = tmp_path / "pool.jsonl"
    ptc = _make_ptc()
    append_to_pool(pool_path, [ptc, ptc])  # same content_hash twice
    stats = dedup_pool(pool_path)
    assert stats["before"] == 2
    assert stats["after"] == 1
    assert stats["removed"] == 1
    loaded = load_pool(pool_path)
    assert len(loaded) == 1


def test_dedup_pool_preserves_distinct_ptcs(tmp_path):
    pool_path = tmp_path / "pool.jsonl"
    ptc_a = _make_ptc("src-001")
    ptc_b = _make_ptc("src-001", "_different")
    append_to_pool(pool_path, [ptc_a, ptc_b])
    stats = dedup_pool(pool_path)
    assert stats["before"] == 2
    assert stats["after"] == 2
    assert stats["removed"] == 0


def test_dedup_pool_missing_file_returns_zero_stats(tmp_path):
    stats = dedup_pool(tmp_path / "none.jsonl")
    assert stats == {"before": 0, "after": 0, "removed": 0}
