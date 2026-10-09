from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from signal_intelligence.ingestion import (
    _extract_date_from_stem,
    _make_slug,
    _make_source_id,
    _strip_html,
    load_documents,
    load_management_docs,
    load_press_docs,
)

# ---------------------------------------------------------------------------
# Unit: date extraction and slug helpers
# ---------------------------------------------------------------------------


def test_extract_date_yyyy_mm_dd():
    assert _extract_date_from_stem("michelin-2024-02-15-annual-report") == date(2024, 2, 15)


def test_extract_date_yyyymmdd():
    assert _extract_date_from_stem("results_20231031") == date(2023, 10, 31)


def test_extract_date_none_when_absent():
    assert _extract_date_from_stem("no-date-here") is None


def test_extract_date_from_text_us_format():
    from signal_intelligence.ingestion import _extract_date_from_text

    text = "PRESS RELEASE\nClermont-Ferrand, February 11, 2026\nAnnual results."
    assert _extract_date_from_text(text) == date(2026, 2, 11)


def test_extract_date_from_text_eu_format():
    from signal_intelligence.ingestion import _extract_date_from_text

    text = "Financial information at September 30, 2025 published 22 October 2025"
    assert _extract_date_from_text(text) == date(2025, 9, 30)


def test_extract_date_from_text_none_when_absent():
    from signal_intelligence.ingestion import _extract_date_from_text

    assert _extract_date_from_text("No dates here at all.") is None


def test_extract_date_from_text_rejects_implausible_years():
    from signal_intelligence.ingestion import _extract_date_from_text

    assert _extract_date_from_text("Founded on May 28, 1889 in Clermont.") is None


def test_make_slug_normalises():
    assert _make_slug("Annual Report 2023") == "annual-report-2023"


def test_make_slug_truncates_to_40():
    long = "a" * 80
    assert len(_make_slug(long)) <= 40


def test_make_slug_removes_special_chars():
    assert _make_slug("résultats_2024!") == "rsultats-2024"


def test_make_source_id_format():
    sid = _make_source_id("ML.PA", "management", date(2024, 2, 15), "annual-report-2023")
    assert sid == "ML.PA-management-2024-02-15-annual-report-2023"


# ---------------------------------------------------------------------------
# Unit: HTML stripping
# ---------------------------------------------------------------------------


def test_strip_html_removes_tags():
    html = "<p>Hello <b>world</b></p>"
    assert _strip_html(html) == "Hello world"


def test_strip_html_removes_script_content():
    html = "<p>Keep this</p><script>remove this</script><p>And this</p>"
    result = _strip_html(html)
    assert "remove this" not in result
    assert "Keep this" in result


def test_strip_html_collapses_whitespace():
    html = "<p>too   many    spaces</p>"
    assert "  " not in _strip_html(html)


# ---------------------------------------------------------------------------
# load_press_docs
# ---------------------------------------------------------------------------


def _make_press_dir(tmp_path: Path, ticker: str = "ML.PA") -> Path:
    press_dir = tmp_path / ticker / "press"
    press_dir.mkdir(parents=True)
    return press_dir


def test_load_press_docs_txt_file(tmp_path):
    press_dir = _make_press_dir(tmp_path)
    (press_dir / "2024-10-15-reuter-article.txt").write_text(
        "Michelin reported strong results.", encoding="utf-8"
    )
    docs = load_press_docs("ML.PA", raw_dir=tmp_path)
    assert len(docs) == 1
    assert docs[0].source_id == "ML.PA-press-2024-10-15-2024-10-15-reuter-article"
    assert docs[0].source_type == "press"
    assert docs[0].source_date == date(2024, 10, 15)
    assert "Michelin" in docs[0].text


def test_load_press_docs_html_file(tmp_path):
    press_dir = _make_press_dir(tmp_path)
    (press_dir / "2024-11-01-article.html").write_text(
        "<html><body><p>Revenue up.</p><script>ignore</script></body></html>",
        encoding="utf-8",
    )
    docs = load_press_docs("ML.PA", raw_dir=tmp_path)
    assert len(docs) == 1
    assert "Revenue up." in docs[0].text
    assert "ignore" not in docs[0].text


def test_load_press_docs_skips_empty_files(tmp_path):
    press_dir = _make_press_dir(tmp_path)
    (press_dir / "2024-10-15-empty.txt").write_text("   ", encoding="utf-8")
    docs = load_press_docs("ML.PA", raw_dir=tmp_path)
    assert docs == []


def test_load_press_docs_missing_dir_returns_empty(tmp_path):
    docs = load_press_docs("ML.PA", raw_dir=tmp_path)
    assert docs == []


def test_load_press_docs_falls_back_to_flat_dir(tmp_path):
    flat_press = tmp_path / "press"
    flat_press.mkdir()
    (flat_press / "2024-10-15-article.txt").write_text("Content here.", encoding="utf-8")
    docs = load_press_docs("ML.PA", raw_dir=tmp_path)
    assert len(docs) == 1


def test_load_press_docs_uses_mtime_when_no_date_in_filename(tmp_path):
    press_dir = _make_press_dir(tmp_path)
    f = press_dir / "undated-article.txt"
    f.write_text("No date in name.", encoding="utf-8")
    docs = load_press_docs("ML.PA", raw_dir=tmp_path)
    assert len(docs) == 1
    assert docs[0].source_date is not None


# ---------------------------------------------------------------------------
# load_management_docs (mocked pdfplumber)
# ---------------------------------------------------------------------------


def _make_mgmt_dir(tmp_path: Path, ticker: str = "ML.PA") -> Path:
    mgmt_dir = tmp_path / ticker / "management"
    mgmt_dir.mkdir(parents=True)
    return mgmt_dir


def test_load_management_docs_parses_pdf(tmp_path):
    mgmt_dir = _make_mgmt_dir(tmp_path)
    pdf_path = mgmt_dir / "2024-02-15-annual-report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Annual report content."
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]

    with patch("signal_intelligence.ingestion.pdfplumber.open", return_value=mock_pdf):
        docs = load_management_docs("ML.PA", raw_dir=tmp_path)

    assert len(docs) == 1
    assert docs[0].source_id == "ML.PA-management-2024-02-15-2024-02-15-annual-report"
    assert docs[0].source_type == "management"
    assert docs[0].source_date == date(2024, 2, 15)
    assert "Annual report content." in docs[0].text


def test_load_management_docs_skips_empty_pdf(tmp_path):
    mgmt_dir = _make_mgmt_dir(tmp_path)
    (mgmt_dir / "2024-02-15-empty.pdf").write_bytes(b"%PDF fake")

    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]

    with patch("signal_intelligence.ingestion.pdfplumber.open", return_value=mock_pdf):
        docs = load_management_docs("ML.PA", raw_dir=tmp_path)

    assert docs == []


def test_load_management_docs_missing_dir_returns_empty(tmp_path):
    docs = load_management_docs("ML.PA", raw_dir=tmp_path)
    assert docs == []


# ---------------------------------------------------------------------------
# load_documents (integration of both loaders)
# ---------------------------------------------------------------------------


def test_load_documents_returns_dicts(tmp_path):
    press_dir = tmp_path / "ML.PA" / "press"
    press_dir.mkdir(parents=True)
    (press_dir / "2024-10-15-article.txt").write_text("Press content.", encoding="utf-8")

    docs = load_documents("ML.PA", raw_dir=tmp_path)
    assert len(docs) == 1
    assert set(docs[0].keys()) == {"source_id", "text", "source_type", "source_date"}


def test_load_documents_source_types_filter(tmp_path):
    press_dir = tmp_path / "ML.PA" / "press"
    press_dir.mkdir(parents=True)
    (press_dir / "2024-10-15-article.txt").write_text("Press content.", encoding="utf-8")

    mgmt_dir = tmp_path / "ML.PA" / "management"
    mgmt_dir.mkdir(parents=True)
    pdf_path = mgmt_dir / "2024-02-15-report.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Management content."
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]

    with patch("signal_intelligence.ingestion.pdfplumber.open", return_value=mock_pdf):
        press_only = load_documents("ML.PA", raw_dir=tmp_path, source_types=["press"])
        mgmt_only = load_documents("ML.PA", raw_dir=tmp_path, source_types=["management"])

    assert all(d["source_type"] == "press" for d in press_only)
    assert all(d["source_type"] == "management" for d in mgmt_only)


def test_load_documents_source_date_is_date_object(tmp_path):
    press_dir = tmp_path / "ML.PA" / "press"
    press_dir.mkdir(parents=True)
    (press_dir / "2024-10-15-article.txt").write_text("Some text.", encoding="utf-8")

    docs = load_documents("ML.PA", raw_dir=tmp_path)
    assert isinstance(docs[0]["source_date"], date)
