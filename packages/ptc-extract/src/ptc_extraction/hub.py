"""Load and save PTC pools on the Hugging Face Hub.

Requires the ``hub`` extra. The package is not on PyPI, so that means installing
``'ptc-extract[hub] @ git+https://github.com/tomPrdd/ptc-extract'`` — see the
README. This module imports without those dependencies present; every function
that needs them raises a clear :class:`ImportError` naming the extra at call
time, so the base package stays importable.

On-Hub layout: one or more ``*.parquet`` files (preferred) or ``*.jsonl`` files
under an optional split directory::

    repo_root/
      train/data.parquet
      validation/data.parquet
      test/data.parquet

A repo with files at the root and no split directories is read whole when
``split`` is None.
"""

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .exceptions import missing_extra
from .schema import PTC

logger = logging.getLogger(__name__)

DEFAULT_POOL_REPO = "tomPrdd/ptc-pool-eu-largecap"
"""The reference pool this library was built to produce.

35,582 typed causal claims from the disclosures and press coverage of five
European large-caps, 2012-2026, with Titan v2 embeddings as a separate config.
"""

_KNOWN_POOLS: dict[str, str] = {
    DEFAULT_POOL_REPO: (
        "Reference pool: 35,582 typed causal claims from European large-cap corporate "
        "disclosures and press (2012-2026), with a separate embeddings config."
    ),
}

# Columns written to parquet, in order. Matches the PTC field order so a parquet
# file and a JSONL pool describe the same thing.
POOL_COLUMNS = (
    "mechanism",
    "raw_text",
    "direction",
    "polarity",
    "source_date",
    "event_date",
    "source_type",
    "source_id",
    "span_start",
    "span_end",
    "extracted_by",
    "extraction_version",
    "confidence",
)


def _require_hub():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise missing_extra("hub", "huggingface_hub", e) from e
    return snapshot_download


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise missing_extra("hub", "pyarrow", e) from e
    return pa, pq


def list_hub_pools() -> dict[str, str]:
    """Curated public PTC pool repo ids, mapped to a one-line description.

    Returns a copy, so callers may not mutate the registry.
    """
    return dict(_KNOWN_POOLS)


def _flatten(ptc: PTC) -> dict[str, Any]:
    """PTC -> flat row. span is split into two nullable integer columns."""
    row = ptc.model_dump(mode="json")
    span = row.pop("span", None)
    row["span_start"] = span[0] if span else None
    row["span_end"] = span[1] if span else None
    return {c: row.get(c) for c in POOL_COLUMNS}


def _unflatten(row: dict[str, Any]) -> dict[str, Any]:
    """Flat row -> PTC-shaped dict.

    Handles both layouts: parquet splits span into span_start/span_end, while a
    JSONL pool written by ``PTC.model_dump_json`` carries a nested ``span``
    list. The parquet columns win when both are present.
    """
    row = dict(row)
    start, end = row.pop("span_start", None), row.pop("span_end", None)
    if start is not None and end is not None:
        row["span"] = (int(start), int(end))
    elif not row.get("span"):
        row["span"] = None
    return {k: v for k, v in row.items() if v is not None or k in {"span", "event_date"}}


def _matches(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        actual = row.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    _, pq = _require_pyarrow()
    table = pq.read_table(path)
    yield from table.to_pylist()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _data_files(root: Path, split: str | None) -> list[Path]:
    """Locate data files, preferring parquet, restricted to a split if given."""
    base = root / split if split else root
    if split and not base.is_dir():
        raise FileNotFoundError(
            f"split {split!r} not found in the downloaded repo. "
            f"Available: {sorted(p.name for p in root.iterdir() if p.is_dir()) or 'none'}"
        )
    for pattern in ("*.parquet", "*.jsonl"):
        files = sorted(base.rglob(pattern))
        if files:
            return files
    raise FileNotFoundError(f"no .parquet or .jsonl data files found under {base}")


def _validate_rows(rows: Iterable[dict[str, Any]], where: str) -> list[PTC]:
    """Validate every row, dropping and counting failures. Never coerces."""
    ptcs: list[PTC] = []
    dropped = 0
    first_error: str | None = None
    for row in rows:
        try:
            ptcs.append(PTC.model_validate(_unflatten(row)))
        except Exception as e:
            dropped += 1
            if first_error is None:
                first_error = str(e).split("\n")[0]
    if dropped:
        logger.warning(
            "Dropped %d of %d rows from %s that failed PTC validation (first error: %s)",
            dropped,
            dropped + len(ptcs),
            where,
            first_error,
        )
    return ptcs


def load_pool_from_hub(
    repo_id: str,
    *,
    revision: str = "main",
    split: str | None = None,
    filters: dict | None = None,
    cache_dir: Path | None = None,
) -> list[PTC]:
    """Download a PTC pool from a Hugging Face dataset repo and validate it.

    Every returned row is validated against the current PTC schema; rows that
    fail validation are dropped and counted, and the count is logged (never
    silently ignored, never coerced).

    ``filters`` supports equality filtering on any PTC field, e.g.
    ``{"source_type": "management", "extraction_version": "v1.1"}``. A list,
    tuple or set as the value means "any of these".

    Requires the ``hub`` extra (see the module docstring; not on PyPI).
    """
    snapshot_download = _require_hub()
    local = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    files = _data_files(local, split)

    rows: list[dict[str, Any]] = []
    for path in files:
        reader = _iter_parquet if path.suffix == ".parquet" else _iter_jsonl
        rows.extend(reader(path))

    if filters:
        rows = [r for r in rows if _matches(r, filters)]

    return _validate_rows(rows, f"{repo_id}@{revision}")


def save_pool_for_hub(
    ptcs: list[PTC],
    path: Path,
    *,
    split: str | None = None,
) -> Path:
    """Write a pool in the on-disk layout :func:`load_pool_from_hub` expects.

    ``path`` is the repo root. Returns the file written. Parquet is used when
    pyarrow is available; the round trip through :func:`load_pool_from_hub` is
    lossless either way.

    Requires the ``hub`` extra for the parquet format.
    """
    root = Path(path)
    target_dir = root / split if split else root
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = [_flatten(p) for p in ptcs]
    pa, pq = _require_pyarrow()

    # Explicit schema so an empty pool still produces a readable, typed file.
    schema = pa.schema(
        [
            ("mechanism", pa.string()),
            ("raw_text", pa.string()),
            ("direction", pa.string()),
            ("polarity", pa.int8()),
            ("source_date", pa.string()),
            ("event_date", pa.string()),
            ("source_type", pa.string()),
            ("source_id", pa.string()),
            ("span_start", pa.int64()),
            ("span_end", pa.int64()),
            ("extracted_by", pa.string()),
            ("extraction_version", pa.string()),
            ("confidence", pa.float64()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    out = target_dir / "data.parquet"
    pq.write_table(table, out)
    return out
