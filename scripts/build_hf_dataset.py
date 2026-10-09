"""Build the Hugging Face dataset artefacts from the deduplicated PTC pool.

Produces two parquet files, deliberately kept apart:

    claims.parquet        one row per claim, ~5 MB
    embeddings.parquet    content_hash + 1024-dim Titan v2 vector, ~160 MB

They are separate because embeddings are a derived, model-specific artefact
(Titan v2, regenerable for ~$0.05) while the claims cost real Bedrock spend and
are permanent. Bundling them would make every consumer download 160 MB for data
most do not want, and would force a full re-upload the day the embedding model
changes. Join on `content_hash`, which is unique only after `dedup_pool()` has
run — verify with the check this script prints.

Copyright: `raw_text` is nulled on every press row. Those are third-party news
articles; the companies' own filings are not affected. `mechanism` is retained
throughout, but press mechanisms that are near-copies of their source passage
are flagged rather than silently shipped.

Usage:
    python scripts/build_hf_dataset.py --pool data/processed/ptc_pool_dataset.jsonl \
        --out data/hf_dataset
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signal_intelligence.ingestion import _extract_date_from_stem, _make_source_id

COMPANIES = {
    "ML.PA": ("Michelin", "Auto components"),
    "OR.PA": ("L'Oréal", "Personal products"),
    "SAN.PA": ("Sanofi", "Pharmaceuticals"),
    "MC.PA": ("LVMH", "Luxury goods"),
    "TTE.PA": ("TotalEnergies", "Oil & gas"),
}

# Aliases used to decide whether a claim actually names its focal company.
ALIASES = {
    "ML.PA": ["michelin", "camso", "bfgoodrich", "uniroyal", "symbio", "tci", "fenner"],
    "OR.PA": [
        "l'oreal",
        "l'oréal",
        "loreal",
        "lancome",
        "lancôme",
        "garnier",
        "maybelline",
        "kiehl",
        "la roche-posay",
        "vichy",
        "aesop",
        "cerave",
    ],
    "SAN.PA": [
        "sanofi",
        "genzyme",
        "dupixent",
        "aubagio",
        "lantus",
        "zentiva",
        "toujeo",
        "beyfortus",
        "praluent",
        "rezurock",
    ],
    "MC.PA": [
        "lvmh",
        "louis vuitton",
        "dior",
        "moet",
        "moët",
        "hennessy",
        "sephora",
        "tiffany",
        "bulgari",
        "fendi",
        "celine",
        "loro piana",
        "givenchy",
        "tag heuer",
    ],
    "TTE.PA": ["total", "totalenergies", "saft", "sunpower", "hutchinson", "maersk oil"],
}

NEAR_DUP_COSINE = 0.95
MECHANISM_COPY_JACCARD = 0.80
_WORD = re.compile(r"\w+", re.UNICODE)


def _ticker(source_id: str) -> str:
    return str(source_id).split("-")[0]


def _tokens(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not (ta | tb):
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _colliding_source_ids(raw_dir: Path) -> set[str]:
    """source_ids that map to more than one file (see CLEANUP_PLAN 2.4b)."""
    colliding: set[str] = set()
    for ticker in COMPANIES:
        for source_type in ("press", "management"):
            d = raw_dir / ticker / source_type
            if not d.is_dir():
                continue
            counts: Counter = Counter()
            for p in sorted(d.glob("*")):
                if p.suffix.lower() not in {".txt", ".pdf", ".html", ".htm"}:
                    continue
                dt = _extract_date_from_stem(p.stem)
                if dt:
                    counts[_make_source_id(ticker, source_type, dt, p.stem)] += 1
            colliding |= {k for k, v in counts.items() if v > 1}
    return colliding


def _docs_naming_focal(raw_dir: Path) -> dict[str, bool]:
    """Does each press document's full text name the company it was fetched for?

    This is the honest topicality test, and it is fully deterministic — no LLM
    needed. Asking whether the *claim* names the company is far too strict: a
    perfectly on-topic claim usually says "the Group expects margins to
    compress" rather than repeating the company name, so that test rejects 83%
    of good rows. The *document* is where the evidence lives — a news article
    genuinely about Michelin names Michelin somewhere.

    Only press is checked. A company's own filing is definitionally about that
    company and needs no test.
    """
    naming: dict[str, bool] = {}
    for ticker, aliases in ALIASES.items():
        d = raw_dir / ticker / "press"
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.txt")):
            dt = _extract_date_from_stem(p.stem)
            if not dt:
                continue
            text = p.read_text(encoding="utf-8", errors="replace").lower()
            naming[_make_source_id(ticker, "press", dt, p.stem)] = any(a in text for a in aliases)
    return naming


def _detect_language(text: str) -> str:
    """Language of the source passage. 'und' when there is too little to go on."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    letters = sum(c.isalpha() for c in text)
    if letters < 20:
        return "und"
    try:
        return detect(text)
    except LangDetectException:
        return "und"


def _find(parent: list[int], a: int) -> int:
    """Union-find root with path compression."""
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _near_dup_groups(rows: list[dict], interim: Path) -> dict[str, int]:
    """Union-find over embedding cosine, within each ticker.

    Cross-ticker similarity mostly reflects generic financial phrasing, so
    grouping is deliberately scoped per company.
    """
    group_of: dict[str, int] = {}
    next_group = 0
    for ticker in COMPANIES:
        npy = interim / f"{ticker}_embeddings.npy"
        idx = interim / f"{ticker}_embedding_index.csv"
        if not (npy.exists() and idx.exists()):
            continue
        import pandas as pd

        index = pd.read_csv(idx)
        hashes = index["content_hash"].tolist()
        keep = {r["content_hash"] for r in rows if _ticker(r["source_id"]) == ticker}
        X = np.load(npy).astype(np.float32)
        if len(X) != len(hashes):
            print(f"  ! {ticker}: embedding/index length mismatch, skipping near-dup groups")
            continue
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9

        parent = list(range(len(X)))
        block = 2048
        for i in range(0, len(X), block):
            sims = X[i : i + block] @ X.T
            for k, row in enumerate(sims):
                gi = i + k
                if gi + 1 >= len(X):
                    continue
                for j in np.nonzero(row[gi + 1 :] >= NEAR_DUP_COSINE)[0]:
                    ra, rb = _find(parent, gi), _find(parent, gi + 1 + int(j))
                    if ra != rb:
                        parent[rb] = ra

        members: dict[int, list[str]] = defaultdict(list)
        for i, h in enumerate(hashes):
            if h in keep:
                members[_find(parent, i)].append(h)
        for group in members.values():
            if len(group) > 1:
                for h in group:
                    group_of[h] = next_group
                next_group += 1
        del X
    return group_of


def _assign_splits(rows: list[dict], seed: int = 20260805) -> dict[str, str]:
    """Document-level split, stratified by (ticker, source_type).

    Grouping by source_id rather than by claim: documents yield 1 to 991 claims,
    so a claim-level split would put near-identical claims from one registration
    document on both sides of the boundary — leakage for anyone fine-tuning an
    extractor on this.
    """
    docs: dict[str, tuple[str, str]] = {}
    for r in rows:
        docs[r["source_id"]] = (_ticker(r["source_id"]), r["source_type"])

    rng = np.random.default_rng(seed)
    split_of_doc: dict[str, str] = {}
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for sid, key in docs.items():
        strata[key].append(sid)

    for sids in strata.values():
        sids = sorted(sids)
        rng.shuffle(sids)
        n = len(sids)
        n_val = max(1, round(0.10 * n)) if n >= 10 else 0
        n_test = max(1, round(0.10 * n)) if n >= 10 else 0
        for i, sid in enumerate(sids):
            if i < n_test:
                split_of_doc[sid] = "test"
            elif i < n_test + n_val:
                split_of_doc[sid] = "validation"
            else:
                split_of_doc[sid] = "train"
    return split_of_doc


def build(pool_path: Path, out_dir: Path, raw_dir: Path, interim: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        json.loads(line)
        for line in pool_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"loaded {len(rows)} claims from {pool_path}")

    # content_hash is the join key to the embeddings file; it must be unique.
    hashes = Counter()
    for r in rows:
        key = f"{r['mechanism']}|{r['direction']}|{r['polarity']}|{r['source_id']}"
        r["content_hash"] = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        hashes[r["content_hash"]] += 1
    dupes = {k: v for k, v in hashes.items() if v > 1}
    if dupes:
        raise SystemExit(
            f"content_hash is not unique ({len(dupes)} collisions). "
            "Run ptc_extraction.io.dedup_pool() on the pool first."
        )
    print(f"  content_hash unique across all {len(rows)} rows — safe join key")

    colliding = _colliding_source_ids(raw_dir)
    doc_names_focal = _docs_naming_focal(raw_dir)
    groups = _near_dup_groups(rows, interim)
    splits = _assign_splits(rows)

    stats: Counter = Counter()
    out_rows = []
    for r in rows:
        ticker = _ticker(r["source_id"])
        company, sector = COMPANIES.get(ticker, ("", ""))
        is_press = r["source_type"] == "press"
        raw_text = r.get("raw_text") or ""

        blob = f"{r['mechanism']} {raw_text}".lower()
        focal = any(a in blob for a in ALIASES.get(ticker, []))
        # Management filings are definitionally about their own company; for
        # press, defer to whether the source article names it (see
        # _docs_naming_focal). This is the column to filter on.
        on_topic = True if not is_press else doc_names_focal.get(r["source_id"], False)

        # Flag press mechanisms that are effectively the source passage.
        near_copy = bool(is_press and _jaccard(r["mechanism"], raw_text) >= MECHANISM_COPY_JACCARD)

        language = _detect_language(raw_text)

        event_date = r.get("event_date")
        event_raw = event_date
        suspect = False
        if event_date:
            if event_date < "2000-01-01":
                stats["event_date_nulled"] += 1
                event_date = None
            elif int(r["source_date"][:4]) - int(event_raw[:4]) > 10:
                suspect = True

        span = r.get("span")
        if is_press:
            stats["press_raw_text_stripped"] += 1
        if near_copy:
            stats["press_mechanism_near_copy"] += 1
        if not focal:
            stats["claim_does_not_name_focal"] += 1
        if is_press and not on_topic:
            stats["press_off_topic_document"] += 1

        out_rows.append(
            {
                "content_hash": r["content_hash"],
                "mechanism": r["mechanism"],
                # Third-party article text is not redistributed.
                "raw_text": None if is_press else raw_text,
                "direction": r["direction"],
                "polarity": int(r["polarity"]),
                "source_date": r["source_date"],
                "event_date": event_date,
                "event_date_raw": event_raw,
                "event_date_suspect": suspect,
                "source_type": r["source_type"],
                "source_id": r["source_id"],
                "span_start": span[0] if span else None,
                "span_end": span[1] if span else None,
                "span_match": r.get("span_match"),
                "extracted_by": r["extracted_by"],
                "extraction_version": r["extraction_version"],
                "confidence": r.get("confidence"),
                "ticker": ticker,
                "company_name": company,
                "sector": sector,
                "language": language,
                "focal_company_mentioned": focal,
                "source_doc_names_focal": on_topic,
                "mechanism_near_copy": near_copy,
                "ambiguous_source": r["source_id"] in colliding,
                "near_dup_group": groups.get(r["content_hash"]),
                "split": splits.get(r["source_id"], "train"),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # The dataset card lives in the repo, not in the build output, so that
    # rebuilding (which clears out_dir) cannot lose it.
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    card = docs_dir / "hf_dataset_card.md"
    if card.exists():
        (out_dir / "README.md").write_text(card.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  copied dataset card from {card}")
    else:
        print(f"  ! dataset card not found at {card}")

    # The card embeds these by relative path, so they have to travel with it.
    figures = sorted((docs_dir / "figures").glob("*.png"))
    if figures:
        (out_dir / "figures").mkdir(exist_ok=True)
        for fig in figures:
            shutil.copy2(fig, out_dir / "figures" / fig.name)
        print(f"  copied {len(figures)} card figures")
    else:
        print("  ! no card figures found; run scripts/make_article_figures.py")

    schema = pa.schema(
        [
            ("content_hash", pa.string()),
            ("mechanism", pa.string()),
            ("raw_text", pa.string()),
            ("direction", pa.string()),
            ("polarity", pa.int8()),
            ("source_date", pa.string()),
            ("event_date", pa.string()),
            ("event_date_raw", pa.string()),
            ("event_date_suspect", pa.bool_()),
            ("source_type", pa.string()),
            ("source_id", pa.string()),
            ("span_start", pa.int64()),
            ("span_end", pa.int64()),
            ("span_match", pa.string()),
            ("extracted_by", pa.string()),
            ("extraction_version", pa.string()),
            ("confidence", pa.float64()),
            ("ticker", pa.string()),
            ("company_name", pa.string()),
            ("sector", pa.string()),
            ("language", pa.string()),
            ("focal_company_mentioned", pa.bool_()),
            ("source_doc_names_focal", pa.bool_()),
            ("mechanism_near_copy", pa.bool_()),
            ("ambiguous_source", pa.bool_()),
            ("near_dup_group", pa.int32()),
            ("split", pa.string()),
        ]
    )
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for split in ("train", "validation", "test"):
        subset = [r for r in out_rows if r["split"] == split]
        pq.write_table(
            pa.Table.from_pylist(subset, schema=schema),
            data_dir / f"{split}.parquet",
            compression="zstd",
        )
        print(f"  wrote data/{split}.parquet  {len(subset)} rows")

    # --- embeddings, as a separate artefact ---
    import pandas as pd

    keep = {r["content_hash"] for r in out_rows}
    vecs, keys = [], []
    for ticker in COMPANIES:
        npy, idx = interim / f"{ticker}_embeddings.npy", interim / f"{ticker}_embedding_index.csv"
        if not (npy.exists() and idx.exists()):
            continue
        index = pd.read_csv(idx)
        X = np.load(npy).astype(np.float32)
        if len(X) != len(index):
            continue
        for h, v in zip(index["content_hash"], X, strict=False):
            if h in keep:
                keys.append(h)
                vecs.append(v)
        del X
    seen, uk, uv = set(), [], []
    for h, v in zip(keys, vecs, strict=False):
        if h not in seen:
            seen.add(h)
            uk.append(h)
            uv.append(v.tolist())
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict(
            {"content_hash": uk, "embedding": uv},
            schema=pa.schema(
                [("content_hash", pa.string()), ("embedding", pa.list_(pa.float32(), 1024))]
            ),
        ),
        emb_dir / "embeddings.parquet",
        compression="zstd",
    )
    print(
        f"  wrote embeddings/embeddings.parquet  {len(uk)} vectors "
        f"({100 * len(uk) / len(out_rows):.1f}% of claims covered)"
    )

    print("\n--- applied ---")
    for k in sorted(stats):
        print(f"  {k:28s} {stats[k]}")
    print(
        f"  {'near_dup_group assigned':28s} {sum(1 for r in out_rows if r['near_dup_group'] is not None)}"
    )
    print(f"  {'ambiguous_source':28s} {sum(1 for r in out_rows if r['ambiguous_source'])}")
    print("  splits:", dict(Counter(r["split"] for r in out_rows)))
    print("  languages:", dict(Counter(r["language"] for r in out_rows).most_common(6)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=Path("data/processed/ptc_pool_dataset.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/hf_dataset"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--interim", type=Path, default=Path("data/interim"))
    args = ap.parse_args()
    build(args.pool, args.out, args.raw, args.interim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
