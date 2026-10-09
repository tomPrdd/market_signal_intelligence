import logging
import re
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

import pdfplumber

logger = logging.getLogger(__name__)

DEFAULT_RAW = Path("data/raw")

_DATE_PATTERNS = [
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),  # YYYY-MM-DD
    re.compile(r"(\d{4})(\d{2})(\d{2})"),  # YYYYMMDD
]
_SLUG_RE = re.compile(r"[\s_]+")

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}
_TEXT_DATE_PATTERNS = [
    # "February 11, 2026" / "February 11 2026"
    re.compile(r"([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})"),
    # "11 February 2026"
    re.compile(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})"),
]


@dataclass
class RawDocument:
    source_id: str
    text: str
    source_type: str
    source_date: date
    path: Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_date_from_stem(stem: str) -> date | None:
    for pattern in _DATE_PATTERNS:
        m = pattern.search(stem)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
    return None


def _extract_date_from_text(text: str, scan_chars: int = 3000) -> date | None:
    """Find the first plausible publication date in the opening of a document.

    Used as a fallback when the filename carries no date (common for IR PDFs
    served from CDNs with hashed names). Only dates between 1995 and one year
    from now are accepted.
    """
    head = text[:scan_chars]
    candidates: list[date] = []
    for pattern in _TEXT_DATE_PATTERNS:
        for m in pattern.finditer(head):
            g = m.groups()
            month_name = g[0] if g[0].isalpha() else g[1]
            day_str = g[1] if g[0].isalpha() else g[0]
            month = _MONTHS.get(month_name.lower())
            if month is None:
                continue
            try:
                candidate = date(int(g[2]), month, int(day_str))
            except ValueError:
                continue
            if 1995 <= candidate.year <= date.today().year + 1:
                candidates.append(candidate)
    return candidates[0] if candidates else None


def _make_slug(stem: str) -> str:
    slug = stem.lower()
    slug = _SLUG_RE.sub("-", slug)
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:40]


def _make_source_id(ticker: str, source_type: str, doc_date: date, stem: str) -> str:
    slug = _make_slug(stem)
    return f"{ticker}-{source_type}-{doc_date.isoformat()}-{slug}"


def _resolve_dir(raw_dir: Path, ticker: str, source_type: str) -> Path:
    """Return the directory to scan, with ticker subfolder taking priority."""
    ticker_subdir = raw_dir / ticker / source_type
    if ticker_subdir.exists():
        return ticker_subdir
    flat_subdir = raw_dir / source_type
    if flat_subdir.exists():
        return flat_subdir
    return ticker_subdir  # caller handles missing dir gracefully


class _HTMLStripper(HTMLParser):
    """Minimal HTML stripper using stdlib — no BeautifulSoup needed."""

    _SKIP_TAGS: ClassVar[set[str]] = {"script", "style", "head", "meta", "link"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        return re.sub(r"\s+", " ", raw).strip()


def _strip_html(html: str) -> str:
    parser = _HTMLStripper()
    parser.feed(html)
    return parser.get_text()


def _pdf_to_text(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_management_docs(
    ticker: str,
    raw_dir: Path = DEFAULT_RAW,
) -> list[RawDocument]:
    """Load PDFs and .txt files from data/raw/{ticker}/management/ (or data/raw/management/).

    PDFs are parsed with pdfplumber; .txt files (e.g. earnings-call transcripts)
    are read as-is. Skips files that produce empty text with a warning.
    """
    scan_dir = _resolve_dir(raw_dir, ticker, "management")
    if not scan_dir.exists():
        logger.warning("Management directory not found: %s", scan_dir)
        return []

    docs: list[RawDocument] = []
    for path in sorted(scan_dir.glob("*")):
        if path.suffix.lower() not in {".pdf", ".txt"}:
            continue

        try:
            if path.suffix.lower() == ".pdf":
                text = _pdf_to_text(path)
            else:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            logger.exception("Failed to parse file: %s", path)
            continue

        if not text.strip():
            logger.warning("Empty text from PDF, skipping: %s", path)
            continue

        doc_date = _extract_date_from_stem(path.stem)
        if doc_date is None:
            doc_date = _extract_date_from_text(text)
            if doc_date is not None:
                logger.info("No date in filename %s, using text date %s", path.name, doc_date)
        if doc_date is None:
            doc_date = date.fromtimestamp(path.stat().st_mtime)
            logger.info("No date in filename or text %s, using mtime %s", path.name, doc_date)

        source_id = _make_source_id(ticker, "management", doc_date, path.stem)
        docs.append(
            RawDocument(
                source_id=source_id,
                text=text,
                source_type="management",
                source_date=doc_date,
                path=path,
            )
        )
        logger.info("Loaded management doc: %s (%d chars)", source_id, len(text))

    return docs


def load_press_docs(
    ticker: str,
    raw_dir: Path = DEFAULT_RAW,
) -> list[RawDocument]:
    """Load all .txt and .html files from data/raw/{ticker}/press/ (or data/raw/press/).

    HTML files are stripped of tags. Plain text files are read as-is.
    """
    scan_dir = _resolve_dir(raw_dir, ticker, "press")
    if not scan_dir.exists():
        logger.warning("Press directory not found: %s", scan_dir)
        return []

    docs: list[RawDocument] = []
    for path in sorted(scan_dir.glob("*")):
        if path.suffix.lower() not in {".txt", ".html", ".htm"}:
            continue

        doc_date = _extract_date_from_stem(path.stem)
        if doc_date is None:
            doc_date = date.fromtimestamp(path.stat().st_mtime)
            logger.info("No date in filename %s, using mtime %s", path.name, doc_date)

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            logger.exception("Failed to read file: %s", path)
            continue

        text = _strip_html(raw) if path.suffix.lower() in {".html", ".htm"} else raw.strip()

        if not text:
            logger.warning("Empty text, skipping: %s", path)
            continue

        source_id = _make_source_id(ticker, "press", doc_date, path.stem)
        docs.append(
            RawDocument(
                source_id=source_id,
                text=text,
                source_type="press",
                source_date=doc_date,
                path=path,
            )
        )
        logger.info("Loaded press doc: %s (%d chars)", source_id, len(text))

    return docs


def load_documents(
    ticker: str,
    raw_dir: Path = DEFAULT_RAW,
    source_types: list[str] | None = None,
) -> list[dict]:
    """Return list[dict] ready for PTCExtractor.extract_from_documents.

    Each dict: {source_id, text, source_type, source_date}.
    source_types: None (both) or a subset of ["management", "press"].
    """
    load_all = source_types is None
    docs: list[RawDocument] = []

    if load_all or "management" in source_types:
        docs.extend(load_management_docs(ticker, raw_dir))

    if load_all or "press" in source_types:
        docs.extend(load_press_docs(ticker, raw_dir))

    return [
        {
            "source_id": d.source_id,
            "text": d.text,
            "source_type": d.source_type,
            "source_date": d.source_date,
        }
        for d in docs
    ]
