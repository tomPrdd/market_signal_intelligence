"""Hugging Face pool loading. The download is always mocked — no network in CI."""

import json
from datetime import date
from pathlib import Path

import pytest

from ptc_extraction import PTC
from ptc_extraction.hub import (
    DEFAULT_POOL_REPO,
    POOL_COLUMNS,
    list_hub_pools,
    load_pool_from_hub,
    save_pool_for_hub,
)

pytest.importorskip("pyarrow", reason="hub extra not installed")


def _ptc(n: int = 0, **overrides) -> PTC:
    base = {
        "mechanism": f"Mechanism number {n} that compresses operating margins.",
        "raw_text": f"Passage number {n} from the underlying source document.",
        "direction": "precursor",
        "polarity": -1,
        "source_date": date(2024, 11, 14),
        "event_date": date(2024, 7, 1),
        "source_type": "management",
        "source_id": f"DOC-{n}",
        "span": (10 * n, 10 * n + 40),
        "extracted_by": "test-model",
        "extraction_version": "v1.1",
        "confidence": 0.9,
    }
    return PTC(**{**base, **overrides})


@pytest.fixture
def fake_download(monkeypatch):
    """Point snapshot_download at a local directory instead of the network."""
    calls = {}

    def _factory(repo_root: Path):
        def _snapshot_download(repo_id, revision=None, repo_type=None, cache_dir=None):
            calls.update(
                repo_id=repo_id, revision=revision, repo_type=repo_type, cache_dir=cache_dir
            )
            return str(repo_root)

        monkeypatch.setattr(
            "ptc_extraction.hub._require_hub", lambda: _snapshot_download, raising=True
        )
        return calls

    return _factory


# --- registry -------------------------------------------------------------------


def test_list_hub_pools_includes_the_reference_pool():
    pools = list_hub_pools()
    assert DEFAULT_POOL_REPO in pools
    owner, _, name = DEFAULT_POOL_REPO.partition("/")
    assert owner and name, "must be a well-formed <owner>/<name> repo id"
    assert "TOFILL" not in DEFAULT_POOL_REPO, "the placeholder must not ship"
    assert pools[DEFAULT_POOL_REPO], "every listed pool needs a description"


def test_list_hub_pools_returns_a_copy():
    list_hub_pools()["x/y"] = "injected"
    assert "x/y" not in list_hub_pools()


# --- round trip -----------------------------------------------------------------


def test_round_trip_preserves_every_field(tmp_path, fake_download):
    original = [_ptc(i) for i in range(5)]
    save_pool_for_hub(original, tmp_path)
    fake_download(tmp_path)

    loaded = load_pool_from_hub("acme/pool")

    assert len(loaded) == len(original)
    by_id = {p.source_id: p for p in loaded}
    for o in original:
        assert by_id[o.source_id] == o


def test_round_trip_preserves_none_span_and_none_event_date(tmp_path, fake_download):
    original = [_ptc(1, span=None, event_date=None, confidence=None)]
    save_pool_for_hub(original, tmp_path)
    fake_download(tmp_path)

    loaded = load_pool_from_hub("acme/pool")
    assert loaded[0].span is None
    assert loaded[0].event_date is None
    assert loaded[0].confidence is None
    assert loaded[0] == original[0]


def test_saved_parquet_has_the_documented_columns(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    out = save_pool_for_hub([_ptc(1)], tmp_path)
    assert out.name == "data.parquet"
    assert tuple(pq.read_table(out).column_names) == POOL_COLUMNS


def test_empty_pool_writes_a_readable_typed_file(tmp_path, fake_download):
    save_pool_for_hub([], tmp_path)
    fake_download(tmp_path)
    assert load_pool_from_hub("acme/pool") == []


# --- splits ---------------------------------------------------------------------


def test_split_selects_only_that_directory(tmp_path, fake_download):
    save_pool_for_hub([_ptc(1), _ptc(2)], tmp_path, split="train")
    save_pool_for_hub([_ptc(3)], tmp_path, split="test")
    fake_download(tmp_path)

    assert len(load_pool_from_hub("acme/pool", split="train")) == 2
    assert [p.source_id for p in load_pool_from_hub("acme/pool", split="test")] == ["DOC-3"]


def test_unknown_split_raises_and_names_what_exists(tmp_path, fake_download):
    save_pool_for_hub([_ptc(1)], tmp_path, split="train")
    fake_download(tmp_path)
    with pytest.raises(FileNotFoundError, match="train"):
        load_pool_from_hub("acme/pool", split="validation")


def test_missing_data_files_raise(tmp_path, fake_download):
    fake_download(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"no \.parquet or \.jsonl"):
        load_pool_from_hub("acme/pool")


# --- jsonl fallback --------------------------------------------------------------


def test_jsonl_repos_are_accepted(tmp_path, fake_download):
    original = _ptc(7)
    (tmp_path / "data.jsonl").write_text(original.model_dump_json() + "\n", encoding="utf-8")
    fake_download(tmp_path)

    loaded = load_pool_from_hub("acme/pool")
    assert loaded == [original]


def test_parquet_is_preferred_over_jsonl(tmp_path, fake_download):
    save_pool_for_hub([_ptc(1)], tmp_path)
    (tmp_path / "data.jsonl").write_text(_ptc(99).model_dump_json() + "\n", encoding="utf-8")
    fake_download(tmp_path)

    assert [p.source_id for p in load_pool_from_hub("acme/pool")] == ["DOC-1"]


# --- filters ---------------------------------------------------------------------


def test_filters_on_a_single_field(tmp_path, fake_download):
    save_pool_for_hub([_ptc(1, source_type="management"), _ptc(2, source_type="press")], tmp_path)
    fake_download(tmp_path)

    loaded = load_pool_from_hub("acme/pool", filters={"source_type": "press"})
    assert [p.source_id for p in loaded] == ["DOC-2"]


def test_filters_combine_conjunctively(tmp_path, fake_download):
    save_pool_for_hub(
        [
            _ptc(1, source_type="press", extraction_version="v1.1"),
            _ptc(2, source_type="press", extraction_version="v1"),
            _ptc(3, source_type="management", extraction_version="v1.1"),
        ],
        tmp_path,
    )
    fake_download(tmp_path)

    loaded = load_pool_from_hub(
        "acme/pool", filters={"source_type": "press", "extraction_version": "v1.1"}
    )
    assert [p.source_id for p in loaded] == ["DOC-1"]


def test_filter_accepts_a_set_of_allowed_values(tmp_path, fake_download):
    save_pool_for_hub([_ptc(i, source_id=f"DOC-{i}") for i in (1, 2, 3)], tmp_path)
    fake_download(tmp_path)

    loaded = load_pool_from_hub("acme/pool", filters={"source_id": ["DOC-1", "DOC-3"]})
    assert sorted(p.source_id for p in loaded) == ["DOC-1", "DOC-3"]


def test_filter_matching_nothing_returns_empty(tmp_path, fake_download):
    save_pool_for_hub([_ptc(1)], tmp_path)
    fake_download(tmp_path)
    assert load_pool_from_hub("acme/pool", filters={"source_type": "nonexistent"}) == []


# --- validation is drop-and-count ------------------------------------------------


def test_invalid_rows_are_dropped_counted_and_logged(tmp_path, fake_download, caplog):
    good = _ptc(1).model_dump_json()
    short_mechanism = json.dumps({**json.loads(good), "mechanism": "no", "source_id": "DOC-BAD"})
    missing_field = json.dumps({"mechanism": "x" * 20, "source_id": "DOC-WORSE"})
    (tmp_path / "data.jsonl").write_text(
        "\n".join([good, short_mechanism, missing_field]) + "\n", encoding="utf-8"
    )
    fake_download(tmp_path)

    with caplog.at_level("WARNING"):
        loaded = load_pool_from_hub("acme/pool")

    assert [p.source_id for p in loaded] == ["DOC-1"]
    assert "Dropped 2 of 3 rows" in caplog.text


def test_valid_rows_are_never_coerced(tmp_path, fake_download):
    """A row with an out-of-domain direction is dropped, not mapped onto a default."""
    bad = json.dumps({**json.loads(_ptc(1).model_dump_json()), "direction": "sideways"})
    (tmp_path / "data.jsonl").write_text(bad + "\n", encoding="utf-8")
    fake_download(tmp_path)
    assert load_pool_from_hub("acme/pool") == []


# --- download arguments ----------------------------------------------------------


def test_revision_and_cache_dir_are_forwarded(tmp_path, fake_download):
    save_pool_for_hub([_ptc(1)], tmp_path)
    calls = fake_download(tmp_path)

    load_pool_from_hub("acme/pool", revision="v2", cache_dir=tmp_path / "cache")

    assert calls["repo_id"] == "acme/pool"
    assert calls["revision"] == "v2"
    assert calls["repo_type"] == "dataset"
    assert calls["cache_dir"] == str(tmp_path / "cache")
