import json
from pathlib import Path

from .schema import PTC


def load_pool(path: Path) -> list[PTC]:
    """Load JSONL pool. Return [] if file does not exist."""
    if not path.exists():
        return []
    ptcs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ptcs.append(PTC.model_validate_json(line))
    return ptcs


def append_to_pool(path: Path, ptcs: list[PTC]) -> int:
    """Append PTCs to JSONL. Create file if missing. Return count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for ptc in ptcs:
            f.write(ptc.model_dump_json() + "\n")
    return len(ptcs)


def existing_source_ids(path: Path) -> set[str]:
    """Return set of source_ids already present in the pool."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            sid = data.get("source_id")
            if sid:
                ids.add(sid)
    return ids


def dedup_pool(path: Path) -> dict:
    """Deduplicate the pool in place by content_hash. Return stats dict."""
    if not path.exists():
        return {"before": 0, "after": 0, "removed": 0}
    ptcs = load_pool(path)
    before = len(ptcs)
    seen: set[str] = set()
    unique: list[PTC] = []
    for ptc in ptcs:
        h = ptc.content_hash()
        if h not in seen:
            seen.add(h)
            unique.append(ptc)
    with path.open("w", encoding="utf-8") as f:
        for ptc in unique:
            f.write(ptc.model_dump_json() + "\n")
    return {"before": before, "after": len(unique), "removed": before - len(unique)}
