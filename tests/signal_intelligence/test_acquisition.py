from pathlib import Path
from unittest.mock import MagicMock, patch

from signal_intelligence.acquisition import (
    _date_part,
    _slug,
    _write_if_new,
    acquire_corpus,
    fetch_ir_pdfs,
    fetch_press_releases,
    fetch_stock_news,
    fetch_transcripts,
    load_universe,
    sync_dir_to_s3,
)

# ---------------------------------------------------------------------------
# Unit: helpers
# ---------------------------------------------------------------------------


def test_slug_normalises():
    assert _slug("Q3 Results: Margin Up!") == "q3-results-margin-up"


def test_slug_empty_falls_back():
    assert _slug("!!!") == "untitled"


def test_date_part_strips_time():
    assert _date_part("2026-01-15 10:30:00") == "2026-01-15"


def test_date_part_none_on_garbage():
    assert _date_part("not a date") is None
    assert _date_part(None) is None


def test_write_if_new_skips_existing(tmp_path):
    path = tmp_path / "doc.txt"
    assert _write_if_new(path, "first") is True
    assert _write_if_new(path, "second") is False
    assert path.read_text() == "first"


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


def test_load_universe_defaults_fmp_symbol(tmp_path):
    cfg = tmp_path / "universe.yaml"
    cfg.write_text(
        "companies:\n"
        "  - {ticker: ML.PA, name: Michelin}\n"
        "  - {ticker: NOVN.SW, name: Novartis, fmp_symbol: NVS}\n"
    )
    companies = load_universe(cfg)
    assert companies[0]["fmp_symbol"] == "ML.PA"
    assert companies[1]["fmp_symbol"] == "NVS"


def test_repo_universe_is_valid():
    companies = load_universe(Path("configs/universe.yaml"))
    assert len(companies) >= 40
    tickers = [c["ticker"] for c in companies]
    assert "ML.PA" in tickers
    assert len(set(tickers)) == len(tickers)
    for c in companies:
        assert c["name"] and c["ticker"]


# ---------------------------------------------------------------------------
# FMP fetchers (mocked)
# ---------------------------------------------------------------------------


@patch("signal_intelligence.acquisition._fmp_get")
def test_fetch_transcripts_writes_dated_files(mock_get, tmp_path):
    mock_get.return_value = [
        {"date": "2025-04-28 14:00:00", "quarter": 1, "content": "CEO: strong quarter."},
        {"date": None, "quarter": 2, "content": "undated — dropped"},
        {"date": "2025-07-25 14:00:00", "quarter": 2, "content": ""},
    ]
    n = fetch_transcripts("ML.PA", "ML.PA", years=[2025], raw_dir=tmp_path)
    assert n == 1
    out = tmp_path / "ML.PA" / "management" / "2025-04-28-earnings-call-q1.txt"
    assert out.read_text() == "CEO: strong quarter."


@patch("signal_intelligence.acquisition._fmp_get")
def test_fetch_transcripts_resume_safe(mock_get, tmp_path):
    mock_get.return_value = [
        {"date": "2025-04-28 14:00:00", "quarter": 1, "content": "CEO: strong quarter."}
    ]
    assert fetch_transcripts("ML.PA", "ML.PA", [2025], tmp_path) == 1
    assert fetch_transcripts("ML.PA", "ML.PA", [2025], tmp_path) == 0


@patch("signal_intelligence.acquisition._fmp_get")
def test_fetch_press_releases_stops_on_empty_page(mock_get, tmp_path):
    mock_get.side_effect = [
        [{"date": "2026-01-15 08:00:00", "title": "Q4 Results", "text": "Revenue grew."}],
        [],
    ]
    n = fetch_press_releases("ML.PA", "ML.PA", raw_dir=tmp_path, max_pages=5)
    assert n == 1
    assert mock_get.call_count == 2
    out = tmp_path / "ML.PA" / "press" / "2026-01-15-pr-q4-results.txt"
    assert out.read_text() == "Q4 Results\n\nRevenue grew."


@patch("signal_intelligence.acquisition._fmp_get")
def test_fetch_stock_news_writes_files(mock_get, tmp_path):
    mock_get.return_value = [
        {"publishedDate": "2026-02-01 09:00:00", "title": "Analyst view", "text": "Tires up."},
    ]
    n = fetch_stock_news("ML.PA", "ML.PA", raw_dir=tmp_path)
    assert n == 1
    files = list((tmp_path / "ML.PA" / "press").glob("*.txt"))
    assert len(files) == 1
    assert files[0].name.startswith("2026-02-01-news-")


@patch("signal_intelligence.acquisition._fmp_get")
def test_fetch_stock_news_handles_http_error(mock_get, tmp_path):
    import requests

    mock_get.side_effect = requests.HTTPError("402 payment required")
    assert fetch_stock_news("ML.PA", "ML.PA", raw_dir=tmp_path) == 0


# ---------------------------------------------------------------------------
# fetch_ir_pdfs size cap
# ---------------------------------------------------------------------------


def _mock_pdf_response(content: bytes, content_length: str | None = None):
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    resp.raise_for_status.return_value = None
    resp.headers = {"content-length": content_length} if content_length else {}
    resp.iter_content.return_value = [content]
    return resp


def test_fetch_ir_pdfs_skips_oversized_via_content_length(tmp_path):
    html = '<a href="report.pdf">Report</a>'
    oversized = _mock_pdf_response(b"%PDF-1.4 small", content_length=str(60 * 1024 * 1024))
    with (
        patch("signal_intelligence.acquisition._get_html", return_value=html),
        patch("signal_intelligence.acquisition.requests.get", return_value=oversized),
    ):
        n = fetch_ir_pdfs("ML.PA", "https://example.com/ir", raw_dir=tmp_path, max_pdf_mb=30)
    assert n == 0
    assert not (tmp_path / "ML.PA" / "management" / "report.pdf").exists()


def test_fetch_ir_pdfs_accepts_normal_sized_pdf(tmp_path):
    html = '<a href="report.pdf">Report</a>'
    small = _mock_pdf_response(b"%PDF-1.4 content", content_length=str(2 * 1024 * 1024))
    with (
        patch("signal_intelligence.acquisition._get_html", return_value=html),
        patch("signal_intelligence.acquisition.requests.get", return_value=small),
    ):
        n = fetch_ir_pdfs("ML.PA", "https://example.com/ir", raw_dir=tmp_path, max_pdf_mb=30)
    assert n == 1
    assert (tmp_path / "ML.PA" / "management" / "report.pdf").read_bytes() == b"%PDF-1.4 content"


def test_fetch_ir_pdfs_skips_oversized_without_content_length(tmp_path):
    html = '<a href="report.pdf">Report</a>'
    big_chunk = b"%PDF-1.4" + b"x" * (2 * 1024 * 1024)
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    resp.raise_for_status.return_value = None
    resp.headers = {}
    resp.iter_content.return_value = [big_chunk] * 20  # 40MB streamed, no content-length header
    with (
        patch("signal_intelligence.acquisition._get_html", return_value=html),
        patch("signal_intelligence.acquisition.requests.get", return_value=resp),
    ):
        n = fetch_ir_pdfs("ML.PA", "https://example.com/ir", raw_dir=tmp_path, max_pdf_mb=1)
    assert n == 0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@patch("signal_intelligence.acquisition.fetch_exa_press", return_value=0)
@patch("signal_intelligence.acquisition.fetch_gdelt_press", return_value=7)
@patch("signal_intelligence.acquisition.fetch_stock_news", return_value=3)
@patch("signal_intelligence.acquisition.fetch_press_releases", return_value=2)
@patch("signal_intelligence.acquisition.fetch_transcripts", return_value=4)
def test_acquire_corpus_aggregates_counts(m_tr, m_pr, m_news, m_gdelt, m_exa, tmp_path):
    stats = acquire_corpus({"ticker": "ML.PA", "name": "Michelin"}, raw_dir=tmp_path)
    assert stats == {
        "ticker": "ML.PA",
        "transcripts": 4,
        "press_releases": 2,
        "news": 3,
        "gdelt": 7,
        "exa": 0,
        "ir_pdfs": 0,
    }


# ---------------------------------------------------------------------------
# S3 sync (mocked boto3)
# ---------------------------------------------------------------------------


@patch("signal_intelligence.acquisition.boto3")
def test_sync_dir_to_s3_skips_unchanged(mock_boto, tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world!")
    (tmp_path / ".gitkeep").write_text("")

    s3 = MagicMock()
    mock_boto.client.return_value = s3
    paginator = MagicMock()
    s3.get_paginator.return_value = paginator
    # a.txt already remote with same size (5 bytes) -> skipped
    paginator.paginate.return_value = [{"Contents": [{"Key": "raw/T/a.txt", "Size": 5}]}]

    n = sync_dir_to_s3(tmp_path, "bucket", "raw/T")
    assert n == 1
    uploaded_keys = [call.args[2] for call in s3.upload_file.call_args_list]
    assert uploaded_keys == ["raw/T/b.txt"]


def test_sync_dir_to_s3_missing_dir_returns_zero(tmp_path):
    assert sync_dir_to_s3(tmp_path / "nope", "bucket", "raw/T") == 0


# ---------------------------------------------------------------------------
# Ingestion integration: transcripts (.txt) are loaded as management docs
# ---------------------------------------------------------------------------


def test_management_txt_docs_are_ingested(tmp_path):
    from signal_intelligence.ingestion import load_management_docs

    mgmt = tmp_path / "ML.PA" / "management"
    mgmt.mkdir(parents=True)
    (mgmt / "2025-04-28-earnings-call-q1.txt").write_text("CEO: strong quarter.")

    docs = load_management_docs("ML.PA", raw_dir=tmp_path)
    assert len(docs) == 1
    assert docs[0].source_type == "management"
    assert docs[0].source_id.startswith("ML.PA-management-2025-04-28")
    assert docs[0].text == "CEO: strong quarter."
