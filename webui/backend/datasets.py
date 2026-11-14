"""Dataset registry for the web UI — curated REAL datasets only.

Synthetic generators (easy/medium/hard/brutal) are kept for unit tests
in tests/ but are NOT registered here so UIL memory stays real-data-only.
"""

from __future__ import annotations
import io
import numpy as np

from figpso_uil.webui.backend.curated_datasets import (
    list_curated,
    load_curated,
    describe_curated,
)

# User-uploaded CSV datasets (real data)
_UPLOADED: dict[str, callable] = {}


def list_datasets() -> list[str]:
    return list_curated() + list(_UPLOADED.keys())


def list_datasets_info() -> list[dict]:
    info = describe_curated()
    for name in _UPLOADED:
        info.append({"name": name, "label": f"Uploaded CSV: {name}", "category": "upload"})
    return info


def load_dataset(name: str, seed: int = 42):
    if name in _UPLOADED:
        return _UPLOADED[name](seed=seed)
    return load_curated(name, seed=seed)


def register_uploaded(name: str, X, y) -> None:
    _UPLOADED[name] = lambda seed=42, _X=X, _y=y: (_X, _y)


def load_csv(content: bytes, label_column: str | int = -1):
    """Parse an uploaded CSV into (X, y)."""
    import csv

    text = content.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise ValueError("empty CSV")

    header = None
    if any(not _is_number(v) for v in rows[0]):
        header = rows[0]
        rows = rows[1:]

    if isinstance(label_column, str) and header is not None:
        label_idx = header.index(label_column)
    elif isinstance(label_column, int):
        label_idx = label_column
    else:
        label_idx = -1

    data = np.array([[float(v) for v in row] for row in rows], dtype=np.float64)
    y = data[:, label_idx].astype(int)
    X = np.delete(data, label_idx, axis=1)
    return X, y


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False
