"""Backfill the null `span` field on an existing PTC pool, at zero LLM cost.

Every PTC in the pool predates span support, so all 35,606 have `span = null`.
The offsets can be recovered offline: `source_id` encodes the document, so the
document is re-read with the same ingestion code that produced the extraction,
and `raw_text` is located inside it.

Only *exact* locations are written. The schema documents `span` as the offsets
of `raw_text`, and downstream code is entitled to assume
`document[start:end] == raw_text`. A passage the model paraphrased, elided or
stitched across a page break cannot satisfy that, so it keeps `span = null`
rather than getting a plausible-looking approximation. This is the same
drop-rather-than-repair rule the extractor follows.

Matching is attempted in three passes, each of which still yields exact offsets
into the re-read document:

  1. verbatim
  2. whitespace-normalised (PDF line wrapping differs from the model's echo)
  3. unicode-normalised (curly quotes, ligatures, non-breaking spaces)

Usage:
    python scripts/backfill_spans.py --dry-run                 # report only
    python scripts/backfill_spans.py --tickers ML.PA OR.PA     # subset
    python scripts/backfill_spans.py --out data/processed/ptc_pool_spanned.jsonl

Re-reading PDFs is the slow part (~2 min per 20 glossy documents); press
documents are plain text and near-instant.
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signal_intelligence.ingestion import load_documents

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("backfill_spans")

POOL = Path("data/processed/ptc_pool.jsonl")

_WS = re.compile(r"\s+")

# Typographic characters a PDF carries that the model echoes back as ASCII.
# Written as escapes rather than literals: the whole point of the table is that
# these characters are confusable, so spelling them out invites the confusion.
_TRANSLATE = str.maketrans(
    {
        "\u2019": "'",  # right single quote
        "\u2018": "'",  # left single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u00a0": " ",  # non-breaking space
        "\ufb01": "fi",  # fi ligature
        "\ufb02": "fl",  # fl ligature
    }
)


def _norm_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _norm_unicode(s: str) -> str:
    return _norm_ws(unicodedata.normalize("NFKC", s.translate(_TRANSLATE)))


def _build_index(text: str, transform) -> tuple[str, list[int]]:
    """Transform `text` char by char, keeping a map back to original offsets.

    Returns (transformed_text, offsets) where offsets[i] is the index in the
    original `text` of transformed_text[i]. This is what lets a match found in
    normalised space be reported as exact offsets in the real document.
    """
    out: list[str] = []
    offsets: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        t = transform(ch)
        if t is None:
            continue
        for c in t:
            if c.isspace():
                if prev_space or not out:
                    continue
                out.append(" ")
                offsets.append(i)
                prev_space = True
            else:
                out.append(c)
                offsets.append(i)
                prev_space = False
    return "".join(out), offsets


def _ws_transform(ch: str) -> str:
    return ch


def _unicode_transform(ch: str) -> str:
    return unicodedata.normalize("NFKC", ch.translate(_TRANSLATE))


def locate(raw_text: str, text: str, indexes: dict) -> tuple[tuple[int, int], str] | None:
    """Return ((start, end), method) for raw_text in text, or None.

    ``method`` is one of:

    - ``verbatim``   — ``text[start:end] == raw_text`` byte for byte.
    - ``whitespace`` — the slice equals raw_text once runs of whitespace are
      collapsed. PDF text wraps lines mid-sentence; the model echoed the
      sentence unwrapped. The offsets are exact, the slice carries newlines.
    - ``unicode``    — as above, plus curly quotes / ligatures / non-breaking
      spaces normalised.

    Only ``verbatim`` satisfies ``document[start:end] == raw_text``. The other
    two locate the passage exactly but the slice is whitespace-equivalent
    rather than identical, which is why the method is recorded rather than
    silently flattened.
    """
    if not raw_text:
        return None

    i = text.find(raw_text)
    if i != -1:
        return (i, i + len(raw_text)), "verbatim"

    for key, method, needle_fn in (
        ("ws", "whitespace", _norm_ws),
        ("unicode", "unicode", _norm_unicode),
    ):
        hay, offsets = indexes[key]
        needle = needle_fn(raw_text)
        if not needle:
            continue
        j = hay.find(needle)
        if j == -1:
            continue
        start = offsets[j]
        # End maps to the character after the last matched one.
        last = j + len(needle) - 1
        end = offsets[last] + 1 if last < len(offsets) else len(text)
        if end > start:
            return (start, end), method
    return None


def backfill(tickers: list[str], pool: Path) -> tuple[list[dict], Counter]:
    rows = [
        json.loads(line) for line in pool.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    by_sid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sid[r["source_id"]].append(r)

    stats: Counter = Counter()
    stats["pool_rows"] = len(rows)

    for ticker in tickers:
        try:
            docs = load_documents(ticker)
        except Exception as e:
            logger.warning("skipping %s: %s", ticker, e)
            continue
        # `ingestion._make_slug` truncates to 40 characters, so two documents
        # published on the same day whose names share a long prefix collapse
        # onto one source_id. When that happens the pool cannot say which of
        # them a claim came from, so no span can be assigned honestly: a match
        # found in one file may be a coincidence and the real passage may live
        # in its sibling. Those rows keep span = null and are counted.
        seen_ids = Counter(d["source_id"] for d in docs)
        ambiguous = {sid for sid, n in seen_ids.items() if n > 1}
        if ambiguous:
            n_rows = sum(len(by_sid.get(sid, ())) for sid in ambiguous)
            print(
                f"  {ticker}: {len(ambiguous)} source_id(s) map to several files "
                f"— {n_rows} row(s) left unspanned (ambiguous provenance)",
                flush=True,
            )
            stats["ambiguous_ids"] += len(ambiguous)
            stats["ambiguous_rows"] += n_rows

        print(f"  {ticker}: re-read {len(docs)} documents", flush=True)

        for doc in docs:
            sid, text = doc["source_id"], doc["text"]
            if sid in ambiguous:
                continue
            targets = by_sid.get(sid)
            if not targets:
                continue
            indexes = {
                "ws": _build_index(text, _ws_transform),
                "unicode": _build_index(text, _unicode_transform),
            }
            for r in targets:
                stats["considered"] += 1
                found = locate(r["raw_text"], text, indexes)
                if found is None:
                    stats[f"miss_{r['source_type']}"] += 1
                    continue
                span, method = found
                # Paranoia: never write a span whose slice is blank, and never
                # write one that claims to be verbatim without being verbatim.
                slice_ = text[span[0] : span[1]]
                if not slice_.strip():
                    stats["rejected_empty"] += 1
                    continue
                if method == "verbatim" and slice_ != r["raw_text"]:
                    stats["rejected_inconsistent"] += 1
                    continue
                r["span"] = list(span)
                r["span_match"] = method
                stats[f"filled_{r['source_type']}"] += 1
                stats[f"method_{method}"] += 1

    stats["filled_total"] = stats["filled_management"] + stats["filled_press"]
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--tickers", nargs="*", default=["ML.PA", "MC.PA", "OR.PA", "SAN.PA", "TTE.PA"])
    ap.add_argument("--out", type=Path, default=None, help="default: alongside the pool, _spanned")
    ap.add_argument("--dry-run", action="store_true", help="report coverage, write nothing")
    args = ap.parse_args()

    if not args.pool.exists():
        print(f"pool not found: {args.pool}", file=sys.stderr)
        return 1

    print(f"backfilling spans from {args.pool}")
    rows, stats = backfill(args.tickers, args.pool)

    considered = max(stats["considered"], 1)
    print("\n--- coverage ---")
    print(f"  pool rows                : {stats['pool_rows']}")
    print(f"  rows whose doc was found : {stats['considered']}")
    for st in ("management", "press"):
        f, m = stats[f"filled_{st}"], stats[f"miss_{st}"]
        if f + m:
            print(f"  {st:11s} filled     : {f} / {f + m}  ({100 * f / (f + m):.1f}%)")
    print(
        f"  TOTAL filled             : {stats['filled_total']} ({100 * stats['filled_total'] / considered:.1f}% of resolvable)"
    )
    print(
        "  by match method          : "
        + ", ".join(f"{m}={stats[f'method_{m}']}" for m in ("verbatim", "whitespace", "unicode"))
    )
    if stats["rejected_empty"]:
        print(f"  rejected (empty slice)   : {stats['rejected_empty']}")
    if stats["rejected_inconsistent"]:
        print(f"  rejected (not verbatim)  : {stats['rejected_inconsistent']}")
    if stats["ambiguous_rows"]:
        print(
            f"  skipped, ambiguous doc   : {stats['ambiguous_rows']} rows across "
            f"{stats['ambiguous_ids']} colliding source_id(s)"
        )

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0

    out = args.out or args.pool.with_name(args.pool.stem + "_spanned.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out} ({len(rows)} rows) — original left untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
