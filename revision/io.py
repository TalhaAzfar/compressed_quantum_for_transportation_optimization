"""Input/output helpers for manifest-driven revision experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def project_root() -> Path:
    """Return the top-level folder that contains this revision package."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path relative to the project root or a supplied base path."""
    path = Path(path)
    if path.is_absolute():
        return path
    if base is None:
        base = project_root()
    return Path(base) / path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for metadata records."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(payload: Any, length: int = 12) -> str:
    """Hash a JSON-serializable payload with stable key ordering."""
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write JSON and create parent directories as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True, default=str)
        handle.write("\n")
    return path


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML manifest.

    YAML support is optional. If PyYAML is unavailable, JSON manifests still
    work and YAML manifests raise a clear ImportError.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json(path)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install PyYAML to read YAML manifests.") from exc
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Manifest {path} must contain a mapping.")
        return data
    raise ValueError(f"Unsupported manifest extension: {path.suffix}")


def write_csv_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: list[str] | None = None,
) -> Path:
    """Write row dictionaries to CSV.

    If fieldnames are omitted, they are inferred from all row keys in first-seen
    order.
    """
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_dataframe(path: str | Path, frame: Any, **to_csv_kwargs: Any) -> Path:
    """Write a pandas DataFrame without requiring pandas at module import time."""
    path = Path(path)
    ensure_dir(path.parent)
    frame.to_csv(path, index=False, **to_csv_kwargs)
    return path


def list_csv_artifacts(source_dir: str | Path) -> list[Path]:
    """List CSV files below a source directory, sorted by path."""
    source_dir = Path(source_dir)
    return sorted(p for p in source_dir.rglob("*.csv") if p.is_file())


def artifact_path(
    *parts: str,
    root: str | Path | None = None,
    create_parent: bool = True,
) -> Path:
    """Build a path below the outputs directory."""
    if root is None:
        root = project_root() / "outputs"
    path = Path(root).joinpath(*parts)
    if create_parent:
        ensure_dir(path.parent)
    return path

