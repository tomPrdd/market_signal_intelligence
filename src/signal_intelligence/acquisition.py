"""Corpus acquisition: fetch management and press documents for a universe of companies.

Sources, in order of reliability:
- IR page PDF harvesting         -> data/raw/{ticker}/management/*.pdf
- GDELT press search (free)      -> data/raw/{ticker}/press/*.txt
- Exa press search (optional)    -> data/raw/{ticker}/press/*.txt
- FMP earnings-call transcripts  -> data/raw/{ticker}/management/*.txt (paid FMP plans only)
- FMP press releases / news      -> data/raw/{ticker}/press/*.txt (paid FMP plans only)

Every fetch is resume-safe: a document whose target file already exists is skipped.
Filenames embed the publication date (YYYY-MM-DD) so ingestion.py can derive source_ids.
All artefacts can be mirrored to S3 with sync_dir_to_s3 / sync_ticker_to_s3.
"""

import logging
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_RAW = Path("data/raw")
DEFAULT_UNIVERSE = Path("configs/universe.yaml")

_FMP_BASE = "https://financialmodelingprep.com/api/v3"
_REQUEST_PAUSE_S = 0.35  # stay polite with FMP rate limits
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_PDF_HREF_RE = re.compile(r"href=[\"']([^\"']+\.pdf[^\"']*)[\"']", re.IGNORECASE)


def _fmp_api_key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise OSError("FMP_API_KEY is not set. Add it to your .env file: FMP_API_KEY=your_key_here")
    return key


def _slug(text: str, max_len: int = 60) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "untitled"


def _date_part(raw: str | None) -> str | None:
    """Extract YYYY-MM-DD from an FMP date string like '2026-01-15 10:00:00'."""
    if not raw:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(raw))
    return m.group(1) if m else None


def _fmp_get(path: str, params: dict | None = None) -> list | dict:
    params = dict(params or {})
    params["apikey"] = _fmp_api_key()
    url = f"{_FMP_BASE}/{path}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(_REQUEST_PAUSE_S)
    return resp.json()


def _write_if_new(path: Path, content: str) -> bool:
    """Write content to path unless it already exists. Returns True if written."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


def load_universe(path: Path = DEFAULT_UNIVERSE) -> list[dict]:
    """Load the company universe from YAML.

    Each entry: ticker (yfinance), fmp_symbol, name, sector, country, ir_url (optional).
    """
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    companies = data["companies"]
    for c in companies:
        c.setdefault("fmp_symbol", c["ticker"])
    return companies


# ---------------------------------------------------------------------------
# FMP fetchers
# ---------------------------------------------------------------------------


def fetch_transcripts(
    ticker: str,
    fmp_symbol: str,
    years: list[int],
    raw_dir: Path = DEFAULT_RAW,
) -> int:
    """Fetch earnings-call transcripts for the given years into management/.

    Returns the number of new files written. Degrades gracefully (logs and
    returns) when the FMP plan does not include transcripts.
    """
    out_dir = raw_dir / ticker / "management"
    written = 0
    for year in years:
        try:
            data = _fmp_get(f"earning_call_transcript/{fmp_symbol}", {"year": year})
        except requests.HTTPError as exc:
            logger.warning("Transcripts unavailable for %s year %s: %s", fmp_symbol, year, exc)
            continue
        if not isinstance(data, list):
            logger.warning("Unexpected transcript payload for %s year %s", fmp_symbol, year)
            continue
        for item in data:
            content = (item.get("content") or "").strip()
            doc_date = _date_part(item.get("date"))
            if not content or not doc_date:
                continue
            quarter = item.get("quarter", "x")
            path = out_dir / f"{doc_date}-earnings-call-q{quarter}.txt"
            if _write_if_new(path, content):
                written += 1
    logger.info("Transcripts %s: %d new files", ticker, written)
    return written


def fetch_press_releases(
    ticker: str,
    fmp_symbol: str,
    raw_dir: Path = DEFAULT_RAW,
    max_pages: int = 5,
) -> int:
    """Fetch company press releases into press/. Returns number of new files."""
    out_dir = raw_dir / ticker / "press"
    written = 0
    for page in range(max_pages):
        try:
            data = _fmp_get(f"press-releases/{fmp_symbol}", {"page": page})
        except requests.HTTPError as exc:
            logger.warning("Press releases unavailable for %s: %s", fmp_symbol, exc)
            break
        if not isinstance(data, list) or not data:
            break
        for item in data:
            text = (item.get("text") or "").strip()
            title = item.get("title") or "press-release"
            doc_date = _date_part(item.get("date"))
            if not text or not doc_date:
                continue
            path = out_dir / f"{doc_date}-pr-{_slug(title)}.txt"
            if _write_if_new(path, f"{title}\n\n{text}"):
                written += 1
    logger.info("Press releases %s: %d new files", ticker, written)
    return written


def fetch_stock_news(
    ticker: str,
    fmp_symbol: str,
    raw_dir: Path = DEFAULT_RAW,
    limit: int = 250,
) -> int:
    """Fetch stock news articles into press/. Returns number of new files."""
    out_dir = raw_dir / ticker / "press"
    try:
        data = _fmp_get("stock_news", {"tickers": fmp_symbol, "limit": limit})
    except requests.HTTPError as exc:
        logger.warning("Stock news unavailable for %s: %s", fmp_symbol, exc)
        return 0
    if not isinstance(data, list):
        return 0
    written = 0
    for item in data:
        text = (item.get("text") or "").strip()
        title = item.get("title") or "news"
        doc_date = _date_part(item.get("publishedDate"))
        if not text or not doc_date:
            continue
        path = out_dir / f"{doc_date}-news-{_slug(title)}.txt"
        if _write_if_new(path, f"{title}\n\n{text}"):
            written += 1
    logger.info("Stock news %s: %d new files", ticker, written)
    return written


# ---------------------------------------------------------------------------
# IR page PDF harvesting
# ---------------------------------------------------------------------------


def _get_html(url: str) -> str:
    """Fetch page HTML with requests; fall back to Playwright for JS-heavy pages."""
    headers = {"User-Agent": "Mozilla/5.0 (signal-intelligence corpus fetcher)"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        html = resp.text
        if _PDF_HREF_RE.search(html):
            return html
    except requests.RequestException as exc:
        logger.info("requests fetch failed for %s (%s), trying Playwright", url, exc)
        html = ""

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning("playwright not installed (uv sync --extra acquisition); using plain HTML")
        return html

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=45_000)
            rendered = page.content()
            browser.close()
        return rendered
    except Exception:
        logger.exception("Playwright fetch failed for %s", url)
        return html


_MAX_IR_PDF_MB = 30  # glossy annual/sustainability reports can run 100MB+ of embedded images;
# pdfplumber chokes on those, so skip anything larger than a normal filing


def fetch_ir_pdfs(
    ticker: str,
    ir_url: str,
    raw_dir: Path = DEFAULT_RAW,
    max_pdfs: int = 10,
    max_pdf_mb: int = _MAX_IR_PDF_MB,
) -> int:
    """Harvest report PDFs linked from an investor-relations page into management/.

    Only PDFs whose link or filename contains a YYYY pattern are kept, and the
    file is saved under its own name (ingestion falls back to mtime when no
    date is present in the filename). PDFs over max_pdf_mb are skipped —
    some IR pages link full glossy annual/sustainability reports (100MB+ of
    embedded images) that are impractical to parse and add little extraction
    value over the leaner financial filings. Returns number of new files.
    """
    out_dir = raw_dir / ticker / "management"
    html = _get_html(ir_url)
    if not html:
        return 0

    max_bytes = max_pdf_mb * 1024 * 1024
    seen: set[str] = set()
    written = 0
    for match in _PDF_HREF_RE.finditer(html):
        href = urljoin(ir_url, match.group(1))
        if href in seen:
            continue
        seen.add(href)
        name = _slug(Path(href.split("?")[0]).stem, max_len=80) + ".pdf"
        path = out_dir / name
        if path.exists():
            continue
        try:
            with requests.get(href, timeout=60, stream=True) as resp:
                resp.raise_for_status()
                content_length = resp.headers.get("content-length")
                if content_length and int(content_length) > max_bytes:
                    logger.info(
                        "Skipping oversized IR PDF (%s MB > %s MB cap): %s",
                        int(content_length) // (1024 * 1024),
                        max_pdf_mb,
                        href,
                    )
                    continue
                chunks = []
                total = 0
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        logger.info("Skipping oversized IR PDF (>%s MB cap): %s", max_pdf_mb, href)
                        chunks = None
                        break
                    chunks.append(chunk)
                if chunks is None:
                    continue
                content = b"".join(chunks)
        except requests.RequestException as exc:
            logger.warning("PDF download failed %s: %s", href, exc)
            continue
        if not content.startswith(b"%PDF"):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        written += 1
        logger.info("Downloaded IR PDF: %s", path.name)
        if written >= max_pdfs:
            break
    logger.info("IR PDFs %s: %d new files", ticker, written)
    return written


# ---------------------------------------------------------------------------
# GDELT press search (free, no API key)
# ---------------------------------------------------------------------------

_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_FINANCE_TERMS = (
    "(earnings OR revenue OR profit OR sales OR outlook OR guidance OR shares OR stock)"
)
_MIN_ARTICLE_CHARS = 500
_GDELT_QUERY_PAUSE_S = 2.0  # GDELT's free tier throttles aggressively; stay well under it
_GDELT_MAX_RETRIES = 3


class _ParagraphExtractor(HTMLParser):
    """Extract <p> text from an article page — crude readability."""

    def __init__(self) -> None:
        super().__init__()
        self._in_p = 0
        self._parts: list[str] = []
        self._current: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "p":
            self._in_p += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._in_p > 0:
            self._in_p -= 1
            para = re.sub(r"\s+", " ", " ".join(self._current)).strip()
            self._current = []
            if len(para) >= 80:  # drop nav/boilerplate fragments
                self._parts.append(para)

    def handle_data(self, data: str) -> None:
        if self._in_p:
            self._current.append(data)

    def get_text(self) -> str:
        return "\n\n".join(self._parts)


def _article_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (signal-intelligence corpus fetcher)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException:
        return ""
    parser = _ParagraphExtractor()
    try:
        parser.feed(resp.text)
    except Exception:
        return ""
    return parser.get_text()


def _quarter_starts(start_year: int) -> list[tuple[str, str]]:
    """(startdatetime, enddatetime) GDELT stamps for each quarter from start_year to now."""
    from datetime import date as _date

    today = _date.today()
    spans = []
    for year in range(start_year, today.year + 1):
        for month in (1, 4, 7, 10):
            q_start = _date(year, month, 1)
            if q_start > today:
                break
            end_month = month + 3
            q_end = _date(year + 1, 1, 1) if end_month > 12 else _date(year, end_month, 1)
            q_end = min(q_end, today)
            spans.append((f"{q_start:%Y%m%d}000000", f"{q_end:%Y%m%d}000000"))
    return spans


def fetch_gdelt_press(
    ticker: str,
    company_name: str,
    raw_dir: Path = DEFAULT_RAW,
    start_year: int = 2019,
    per_quarter: int = 20,
) -> int:
    """Fetch finance-related press articles about the company via GDELT into press/.

    Queries one quarter at a time (GDELT date filters) to build history, then
    downloads each article and keeps the <p>-extracted text when substantial.
    Returns the number of new files written.
    """
    out_dir = raw_dir / ticker / "press"
    headers = {"User-Agent": "Mozilla/5.0 (signal-intelligence corpus fetcher)"}
    written = 0

    for start_dt, end_dt in _quarter_starts(start_year):
        params = {
            "query": f'"{company_name}" {_GDELT_FINANCE_TERMS} sourcelang:english',
            "mode": "artlist",
            "format": "json",
            "maxrecords": per_quarter,
            "startdatetime": start_dt,
            "enddatetime": end_dt,
        }
        articles = []
        for attempt in range(_GDELT_MAX_RETRIES):
            try:
                resp = requests.get(_GDELT_URL, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                articles = resp.json().get("articles", [])
                break
            except (requests.RequestException, ValueError) as exc:
                backoff = _GDELT_QUERY_PAUSE_S * (2**attempt)
                logger.warning(
                    "GDELT query failed for %s (%s), attempt %d/%d, backing off %.0fs: %s",
                    ticker,
                    start_dt,
                    attempt + 1,
                    _GDELT_MAX_RETRIES,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
        else:
            logger.warning(
                "GDELT query gave up for %s (%s) after %d attempts",
                ticker,
                start_dt,
                _GDELT_MAX_RETRIES,
            )
        time.sleep(_GDELT_QUERY_PAUSE_S)

        for art in articles:
            seendate = art.get("seendate", "")  # e.g. 20240205T130000Z
            m = re.match(r"(\d{4})(\d{2})(\d{2})", seendate)
            if not m:
                continue
            doc_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            title = art.get("title") or "article"
            path = out_dir / f"{doc_date}-gdelt-{_slug(title)}.txt"
            if path.exists():
                continue
            text = _article_text(art.get("url", ""))
            if len(text) < _MIN_ARTICLE_CHARS:
                continue
            if _write_if_new(path, f"{title}\n\n{text}"):
                written += 1

    logger.info("GDELT press %s: %d new files", ticker, written)
    return written


# ---------------------------------------------------------------------------
# Exa press search (optional — requires EXA_API_KEY)
# ---------------------------------------------------------------------------


def fetch_exa_press(
    ticker: str,
    company_name: str,
    raw_dir: Path = DEFAULT_RAW,
    num_results: int = 50,
) -> int:
    """Fetch press articles about the company via Exa into press/.

    Returns 0 (with a log line) when exa-py or EXA_API_KEY is missing.
    """
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        logger.info("EXA_API_KEY not set; skipping Exa press fetch for %s", ticker)
        return 0
    try:
        from exa_py import Exa
    except ImportError:
        logger.warning("exa-py not installed (uv sync --extra acquisition); skipping Exa fetch")
        return 0

    exa = Exa(api_key)
    out_dir = raw_dir / ticker / "press"
    written = 0
    try:
        results = exa.search_and_contents(
            f"{company_name} company news financial results",
            num_results=num_results,
            text=True,
            category="news",
        )
    except Exception:
        logger.exception("Exa search failed for %s", ticker)
        return 0

    for item in results.results:
        text = (item.text or "").strip()
        doc_date = _date_part(getattr(item, "published_date", None))
        if not text or not doc_date:
            continue
        path = out_dir / f"{doc_date}-exa-{_slug(item.title or 'article')}.txt"
        if _write_if_new(path, f"{item.title}\n\n{text}"):
            written += 1
    logger.info("Exa press %s: %d new files", ticker, written)
    return written


# ---------------------------------------------------------------------------
# S3 sync
# ---------------------------------------------------------------------------


def default_bucket_name() -> str:
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"signal-intelligence-{account_id}"


def ensure_bucket(bucket: str, region: str = "eu-west-3") -> None:
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
    except s3.exceptions.ClientError:
        logger.info("Creating S3 bucket %s in %s", bucket, region)
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )


def sync_dir_to_s3(local_dir: Path, bucket: str, prefix: str, region: str = "eu-west-3") -> int:
    """Upload files under local_dir to s3://bucket/prefix/, skipping unchanged sizes.

    Returns number of files uploaded.
    """
    if not local_dir.exists():
        return 0
    s3 = boto3.client("s3", region_name=region)

    remote_sizes: dict[str, int] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            remote_sizes[obj["Key"]] = obj["Size"]

    uploaded = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        key = f"{prefix}/{path.relative_to(local_dir)}"
        if remote_sizes.get(key) == path.stat().st_size:
            continue
        s3.upload_file(str(path), bucket, key)
        uploaded += 1
    logger.info("S3 sync %s -> s3://%s/%s: %d files uploaded", local_dir, bucket, prefix, uploaded)
    return uploaded


def sync_ticker_to_s3(
    ticker: str,
    bucket: str | None = None,
    raw_dir: Path = DEFAULT_RAW,
    processed_dir: Path = Path("data/processed"),
    market_dir: Path = Path("data/market"),
    region: str = "eu-west-3",
) -> dict:
    """Mirror a ticker's raw corpus, market data, and processed outputs to S3."""
    bucket = bucket or default_bucket_name()
    ensure_bucket(bucket, region)
    n_raw = sync_dir_to_s3(raw_dir / ticker, bucket, f"raw/{ticker}", region)
    n_processed = 0
    if processed_dir.exists():
        s3 = boto3.client("s3", region_name=region)
        for path in sorted(processed_dir.glob(f"{ticker}_*")):
            s3.upload_file(str(path), bucket, f"processed/{ticker}/{path.name}")
            n_processed += 1
        for path in sorted(market_dir.glob(f"{ticker}_*")) if market_dir.exists() else []:
            s3.upload_file(str(path), bucket, f"market/{path.name}")
            n_processed += 1
        pool_path = processed_dir / "ptc_pool.jsonl"
        if pool_path.exists():
            s3.upload_file(str(pool_path), bucket, "processed/ptc_pool.jsonl")
            n_processed += 1
        _update_manifest(s3, bucket)
    return {"bucket": bucket, "raw_uploaded": n_raw, "processed_uploaded": n_processed}


def _update_manifest(s3, bucket: str) -> None:
    """Regenerate processed/manifest.csv — the deployed app's ticker list."""
    tickers: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="processed/"):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) == 3 and parts[2].endswith("_canonicals.csv"):
                tickers.add(parts[1])
    body = "ticker\n" + "\n".join(sorted(tickers)) + "\n"
    s3.put_object(Bucket=bucket, Key="processed/manifest.csv", Body=body.encode())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def acquire_corpus(
    company: dict,
    raw_dir: Path = DEFAULT_RAW,
    years: list[int] | None = None,
    max_news: int = 250,
    max_ir_pdfs: int = 10,
) -> dict:
    """Fetch all available sources for one company. Returns per-source counts.

    company: dict with keys ticker, fmp_symbol, name, and optionally ir_url.
    """
    ticker = company["ticker"]
    fmp_symbol = company.get("fmp_symbol", ticker)
    years = years or list(range(2019, 2027))

    stats = {
        "ticker": ticker,
        # IR PDFs and FMP first: fast, independent of GDELT's rate limiting.
        "ir_pdfs": 0,
        "transcripts": fetch_transcripts(ticker, fmp_symbol, years, raw_dir),
        "press_releases": fetch_press_releases(ticker, fmp_symbol, raw_dir),
        "news": fetch_stock_news(ticker, fmp_symbol, raw_dir, limit=max_news),
    }
    if company.get("ir_url"):
        stats["ir_pdfs"] = fetch_ir_pdfs(ticker, company["ir_url"], raw_dir, max_pdfs=max_ir_pdfs)
    # GDELT last: slowest source, most prone to rate limiting.
    stats["gdelt"] = fetch_gdelt_press(ticker, company.get("name", ticker), raw_dir)
    stats["exa"] = fetch_exa_press(ticker, company.get("name", ticker), raw_dir)

    logger.info("Corpus acquisition complete for %s: %s", ticker, stats)
    return stats
