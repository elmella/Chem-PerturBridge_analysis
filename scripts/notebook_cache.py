"""Resumable notebook stages.

These notebooks run long scoring loops and bootstrap passes whose results are written to
TSV anyway. Wrapping each expensive stage in :func:`cached_frame` makes a notebook
restartable: a stage whose output file already exists is reloaded instead of recomputed,
so a crash late in a notebook no longer means redoing everything above it.

Usage in a notebook::

    from scripts.notebook_cache import cache_summary, cached_frame, force_recompute

    def build_matched_pair_metrics() -> pd.DataFrame:
        ...                                   # the expensive work
        return frame

    path = OUTPUT_DIR / "matched_sample_pair_metrics.tsv"
    matched_pair_metrics = cached_frame("deg_metrics", path, build_matched_pair_metrics)

To rebuild a stage, delete its TSV or name it explicitly::

    force_recompute("peer_baselines")         # in a cell, before the stage runs
    CPB_FORCE_RECOMPUTE=peer_baselines,peer_ci jupyter lab      # or from the shell
    CPB_FORCE_RECOMPUTE=all jupyter lab                        # ignore every cache

Prefer :func:`force_recompute` over rebinding ``FORCE_RECOMPUTE``: the set is shared by
reference, so reassigning the name in a notebook would silently detach it.

Run ``python scripts/notebook_cache.py`` to execute the self-tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TextIO

import pandas as pd

__all__ = [
    "CACHE_STATUS",
    "CACHE_STRING_COLUMNS",
    "FORCE_RECOMPUTE",
    "FORCE_ALL",
    "cache_summary",
    "cached_frame",
    "IncrementalContextFrameStore",
    "file_inventory",
    "file_inventory_fingerprint",
    "format_progress",
    "force_recompute",
    "frame_identity_fingerprint",
    "is_cached",
    "ProgressReporter",
    "resumable_context_frame",
    "stable_json_fingerprint",
]

# Read back with inferred dtypes, a key like "00123" becomes the integer 123 and silently
# stops matching the string keys used for row lookups -- which surfaces as mass unresolved
# rows rather than an error. These columns are therefore always reloaded as strings;
# everything else keeps its inferred dtype.
CACHE_STRING_COLUMNS: tuple[str, ...] = (
    "dataset_a",
    "dataset_b",
    "query_dataset",
    "target_dataset",
    "dataset_name",
    "cell_type",
    "time_key",
    "dose_key",
    "pubchem_cid",
    "query_pubchem_cid",
    "condition_key",
    "obs_id",
    "query_obs_id",
    "left_obs_id",
    "right_obs_id",
    "left_dose_key",
    "right_dose_key",
    "query_dose_key",
    "best_target_dose_key",
    "left_plate",
    "right_plate",
    "left_well",
    "right_well",
    "matched_condition_key",
    "left_adj_pvalue_layer",
    "right_adj_pvalue_layer",
    "perturbagen_display",
    "dose_threshold",
    "representation",
    "retrieval_variant",
    "similarity_metric",
    "baseline_type",
    "baseline_role",
    "metric",
    "value_col",
    "summary_level",
    "cluster_col",
    "inner_strata",
    "outer_strata",
    "uncertainty_scope",
    "ci_method",
    "ci_status",
    "direction",
)

_ENV_FORCE = os.environ.get("CPB_FORCE_RECOMPUTE", "")
FORCE_ALL: bool = _ENV_FORCE.strip().lower() == "all"
FORCE_RECOMPUTE: set[str] = (
    set()
    if FORCE_ALL
    else {stage.strip() for stage in _ENV_FORCE.split(",") if stage.strip()}
)
CACHE_STATUS: dict[str, str] = {}


def _json_ready(value: Any) -> Any:
    """Convert common analysis objects into a deterministic JSON-compatible value."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_json_ready(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    # NumPy scalar types, pandas scalar wrappers, and enums commonly expose item().
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError):
            pass
        else:
            if scalar is not value:
                return _json_ready(scalar)
    # Timestamps and similar scalar objects have a stable ISO representation.
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except (TypeError, ValueError):
            pass
    raise TypeError(
        f"Cannot create a stable JSON fingerprint for {type(value).__name__}"
    )


def stable_json_fingerprint(payload: Any) -> str:
    """Return a SHA-256 fingerprint of a canonical JSON representation.

    Mapping and set order do not affect the result. Paths, NumPy scalars, timestamps,
    and non-finite floats are normalized explicitly instead of relying on ``repr``.
    """
    serialized = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def file_inventory(
    paths: Iterable[Path],
    *,
    root: Optional[Path] = None,
    include_content_hash: bool = False,
) -> list[dict[str, Any]]:
    """Describe input files deterministically for inclusion in a cache fingerprint.

    By default this records path, size, and nanosecond modification time, which is fast
    even for large H5AD inputs. ``include_content_hash=True`` is available for small
    inputs when a byte-for-byte identity is required.
    """
    root_path = Path(root).resolve() if root is not None else None
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        resolved = path.resolve()
        stat = resolved.stat()
        if not resolved.is_file():
            raise ValueError(f"Inventory path is not a regular file: {path}")
        if root_path is None:
            display_path = str(resolved)
        else:
            try:
                display_path = str(resolved.relative_to(root_path))
            except ValueError:
                display_path = str(resolved)
        record: dict[str, Any] = {
            "path": display_path,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        if include_content_hash:
            digest = hashlib.sha256()
            with resolved.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            record["sha256"] = digest.hexdigest()
        records.append(record)
    return sorted(records, key=lambda record: str(record["path"]))


def file_inventory_fingerprint(
    paths: Iterable[Path],
    *,
    root: Optional[Path] = None,
    payload: Any = None,
    include_content_hash: bool = False,
) -> str:
    """Fingerprint a file inventory together with optional analysis configuration."""
    return stable_json_fingerprint(
        {
            "files": file_inventory(
                paths,
                root=root,
                include_content_hash=include_content_hash,
            ),
            "payload": payload,
        }
    )


def frame_identity_fingerprint(
    frame: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    *,
    order_sensitive: bool = False,
) -> str:
    """Fingerprint selected DataFrame row identities without serializing a large TSV.

    The default treats rows as a multiset, which is appropriate for validating that a
    cached analysis covers the same matched pairs even if their iteration order changes.
    Set ``order_sensitive=True`` when row order is part of the stage contract.
    """
    selected_columns = list(frame.columns if columns is None else columns)
    missing_columns = set(selected_columns) - set(frame.columns)
    if missing_columns:
        raise KeyError(f"Frame identity columns are missing: {sorted(missing_columns)!r}")
    selected = frame.loc[:, selected_columns]
    row_hashes = pd.util.hash_pandas_object(
        selected,
        index=False,
        categorize=True,
    ).to_numpy(dtype="<u8", copy=True)
    if not order_sensitive:
        row_hashes.sort()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": selected_columns,
                "n_rows": int(len(selected)),
                "order_sensitive": bool(order_sensitive),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_progress(
    *,
    label: str,
    completed: int,
    total: Optional[int],
    elapsed_seconds: float,
    detail: Optional[str] = None,
) -> str:
    """Format elapsed time, throughput, and ETA for a long notebook loop."""
    completed = max(0, int(completed))
    elapsed_seconds = max(0.0, float(elapsed_seconds))
    rate = completed / elapsed_seconds if completed > 0 and elapsed_seconds > 0 else None
    if total is None:
        count_text = f"{completed:,}"
        eta = None
    else:
        total = max(0, int(total))
        percent = (100.0 * completed / total) if total else 100.0
        count_text = f"{completed:,}/{total:,} ({percent:.1f}%)"
        eta = (
            max(0, total - completed) / rate
            if rate is not None and rate > 0
            else None
        )
    rate_text = f"{rate:,.2f}/s" if rate is not None else "--/s"
    message = (
        f"[{label}] {count_text} | elapsed {_format_duration(elapsed_seconds)}"
        f" | {rate_text} | ETA {_format_duration(eta)}"
    )
    if detail:
        message = f"{message} | {detail}"
    return message


@dataclass
class ProgressReporter:
    """Rate-limited progress output for long-running notebook loops."""

    total: Optional[int] = None
    label: str = "progress"
    every: int = 25
    min_interval_seconds: float = 30.0
    stream: Optional[TextIO] = None
    time_fn: Callable[[], float] = time.monotonic
    completed: int = 0
    _started_at: float = field(init=False, repr=False)
    _last_report_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.total is not None and self.total < 0:
            raise ValueError("total must be non-negative")
        if self.every < 1:
            raise ValueError("every must be at least one")
        if self.min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        now = float(self.time_fn())
        self._started_at = now
        self._last_report_at = now

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, float(self.time_fn()) - self._started_at)

    def format(self, *, detail: Optional[str] = None) -> str:
        return format_progress(
            label=self.label,
            completed=self.completed,
            total=self.total,
            elapsed_seconds=self.elapsed_seconds,
            detail=detail,
        )

    def update(
        self,
        advance: int = 1,
        *,
        completed: Optional[int] = None,
        detail: Optional[str] = None,
        force: bool = False,
    ) -> Optional[str]:
        """Advance progress and print when the count/time threshold is reached."""
        if completed is None:
            self.completed += int(advance)
        else:
            self.completed = int(completed)
        now = float(self.time_fn())
        is_complete = self.total is not None and self.completed >= self.total
        count_due = self.completed % self.every == 0
        time_due = now - self._last_report_at >= self.min_interval_seconds
        if not (force or is_complete or count_due or time_due):
            return None
        message = format_progress(
            label=self.label,
            completed=self.completed,
            total=self.total,
            elapsed_seconds=max(0.0, now - self._started_at),
            detail=detail,
        )
        print(message, file=self.stream or sys.stdout, flush=True)
        self._last_report_at = now
        return message


def force_recompute(*stages: str, replace: bool = False) -> set[str]:
    """Mark stages to recompute even when their output file exists.

    Mutates the shared set rather than rebinding it, so notebook cells and this module
    stay in agreement. Pass ``replace=True`` to drop any previous selection.
    """
    if replace:
        FORCE_RECOMPUTE.clear()
    FORCE_RECOMPUTE.update(str(stage) for stage in stages)
    return set(FORCE_RECOMPUTE)


def is_cached(
    stage: str,
    path: Path,
    *,
    fingerprint: Optional[str] = None,
) -> bool:
    """Whether :func:`cached_frame` would reload this stage rather than rebuild it.

    Useful when the expensive work is a module-level loop that would be awkward to move
    into a builder: guard the loop's input with this so it becomes a no-op on a cache hit,
    and let ``cached_frame`` assemble the result from whatever the loop produced.
    """
    path = Path(path)
    if not path.exists() or FORCE_ALL or stage in FORCE_RECOMPUTE:
        return False
    if fingerprint is None:
        return True
    metadata_path = path.with_name(f"{path.name}.cache.json")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return metadata.get("fingerprint") == str(fingerprint)


def cached_frame(
    stage: str,
    path: Path,
    build: Callable[[], pd.DataFrame],
    *,
    string_columns: Optional[Sequence[str]] = None,
    fingerprint: Optional[str] = None,
    required_columns: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reload ``path`` when it exists, otherwise run ``build()`` and save the result.

    ``stage`` is the name used by :func:`force_recompute` and reported in
    :data:`CACHE_STATUS`.
    """
    path = Path(path)
    columns: Iterable[str] = (
        CACHE_STRING_COLUMNS if string_columns is None else tuple(string_columns)
    )
    metadata_path = path.with_name(f"{path.name}.cache.json")
    cache_is_compatible = is_cached(
        stage,
        path,
        fingerprint=fingerprint,
    )

    if cache_is_compatible:
        frame = pd.read_csv(
            path,
            sep="\t",
            dtype={column_name: str for column_name in columns},
        )
        for column_name in columns:
            if column_name in frame.columns:
                frame[column_name] = (
                    frame[column_name].astype("string").fillna("").astype(str)
                )
        missing_columns = set(required_columns or ()) - set(frame.columns)
        if missing_columns:
            cache_is_compatible = False
        else:
            CACHE_STATUS[stage] = "reloaded"
            if verbose:
                print(f"[{stage}] reloaded {len(frame):,} rows from {path.name}")
            return frame

    frame = build()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"[{stage}] build() returned {type(frame).__name__}, not a DataFrame")
    missing_columns = set(required_columns or ()) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"[{stage}] build() omitted required columns: {sorted(missing_columns)!r}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary_metadata_path = metadata_path.with_name(
        f".{metadata_path.name}.tmp-{os.getpid()}"
    )
    try:
        frame.to_csv(temporary_path, sep="\t", index=False)
        os.replace(temporary_path, path)
        if fingerprint is not None:
            temporary_metadata_path.write_text(
                json.dumps(
                    {
                        "fingerprint": str(fingerprint),
                        "columns": frame.columns.tolist(),
                        "n_rows": int(len(frame)),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            os.replace(temporary_metadata_path, metadata_path)
    finally:
        for temporary in (temporary_path, temporary_metadata_path):
            if temporary.exists():
                temporary.unlink()
    CACHE_STATUS[stage] = "computed"
    if verbose:
        print(f"[{stage}] computed and saved {len(frame):,} rows to {path.name}")
    return frame


def _read_tsv(
    path: Path,
    *,
    string_columns: Sequence[str],
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if columns is not None and len(columns) == 0:
        return pd.DataFrame()
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype={column_name: str for column_name in string_columns},
    )
    for column_name in string_columns:
        if column_name in frame.columns:
            frame[column_name] = (
                frame[column_name].astype("string").fillna("").astype(str)
            )
    return frame


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        temporary_path.write_text(text)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        frame.to_csv(temporary_path, sep="\t", index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class IncrementalContextFrameStore:
    """Atomic context shards for notebook loops that cannot be callbacks.

    ``resumable_context_frame`` remains the preferred interface. This adapter exists
    for large notebook cells whose metric implementation is an established top-level
    context loop: callers skip completed keys, save the rows produced by each context,
    and assemble all shards before publishing the conventional stage TSV.
    """

    def __init__(
        self,
        stage: str,
        shard_root: Path,
        context_keys: Iterable[Any],
        *,
        fingerprint: str,
        string_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.stage = str(stage)
        self.fingerprint = str(fingerprint)
        self.string_columns = (
            CACHE_STRING_COLUMNS
            if string_columns is None
            else tuple(string_columns)
        )
        self.force = FORCE_ALL or self.stage in FORCE_RECOMPUTE
        self.keys = list(context_keys)
        self._key_hashes: dict[str, Any] = {}
        for key in self.keys:
            key_hash = stable_json_fingerprint(key)
            if key_hash in self._key_hashes:
                raise ValueError(
                    f"[{self.stage}] duplicate context key: {key!r}; "
                    f"first seen as {self._key_hashes[key_hash]!r}"
                )
            self._key_hashes[key_hash] = key

        self.run_fingerprint = stable_json_fingerprint(
            {
                "stage": self.stage,
                "fingerprint": self.fingerprint,
                "context_key_hashes": sorted(self._key_hashes),
            }
        )
        self.run_directory = Path(shard_root) / self.run_fingerprint
        self.run_directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.run_directory / "manifest.json",
            json.dumps(
                {
                    "stage": self.stage,
                    "fingerprint": self.fingerprint,
                    "run_fingerprint": self.run_fingerprint,
                    "n_contexts": len(self.keys),
                    "context_key_hashes": sorted(self._key_hashes),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def _paths(self, key: Any) -> tuple[str, Path, Path]:
        key_hash = stable_json_fingerprint(key)
        if key_hash not in self._key_hashes:
            raise KeyError(f"[{self.stage}] unknown context key: {key!r}")
        return (
            key_hash,
            self.run_directory / f"{key_hash}.tsv",
            self.run_directory / f"{key_hash}.cache.json",
        )

    def _metadata(self, key: Any) -> Optional[dict[str, Any]]:
        key_hash, shard_path, metadata_path = self._paths(key)
        if not shard_path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if (
            metadata.get("stage") != self.stage
            or metadata.get("fingerprint") != self.fingerprint
            or metadata.get("run_fingerprint") != self.run_fingerprint
            or metadata.get("context_key_hash") != key_hash
            or not isinstance(metadata.get("columns"), list)
        ):
            return None
        return metadata

    def is_complete(self, key: Any) -> bool:
        """Return whether ``key`` has a compatible atomic shard."""
        return not self.force and self._metadata(key) is not None

    def save(self, key: Any, frame: pd.DataFrame) -> None:
        """Atomically replace the shard for ``key``."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"[{self.stage}] context {key!r} returned "
                f"{type(frame).__name__}, not a DataFrame"
            )
        key_hash, shard_path, metadata_path = self._paths(key)
        _atomic_write_frame(shard_path, frame)
        _atomic_write_text(
            metadata_path,
            json.dumps(
                {
                    "stage": self.stage,
                    "fingerprint": self.fingerprint,
                    "run_fingerprint": self.run_fingerprint,
                    "context_key": _json_ready(key),
                    "context_key_hash": key_hash,
                    "columns": frame.columns.tolist(),
                    "n_rows": int(len(frame)),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def assemble(self) -> pd.DataFrame:
        """Load all completed shards in the caller's context order."""
        frames: list[pd.DataFrame] = []
        expected_columns: Optional[list[str]] = None
        for key in self.keys:
            metadata = self._metadata(key)
            if metadata is None:
                raise RuntimeError(
                    f"[{self.stage}] context {key!r} has not been checkpointed"
                )
            _, shard_path, _ = self._paths(key)
            columns = metadata["columns"]
            try:
                frame = _read_tsv(
                    shard_path,
                    string_columns=self.string_columns,
                    columns=columns,
                )
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                raise RuntimeError(
                    f"[{self.stage}] could not reload context {key!r}"
                ) from exc
            if frame.columns.tolist() != columns or len(frame) != int(
                metadata.get("n_rows", -1)
            ):
                raise RuntimeError(
                    f"[{self.stage}] context {key!r} failed shard validation"
                )
            if columns:
                if expected_columns is None:
                    expected_columns = columns
                elif columns != expected_columns:
                    raise RuntimeError(
                        f"[{self.stage}] context {key!r} has columns {columns!r}; "
                        f"expected {expected_columns!r}"
                    )
            frames.append(frame)

        nonempty_schema_frames = [frame for frame in frames if len(frame.columns) > 0]
        if not nonempty_schema_frames:
            return pd.DataFrame()
        return pd.concat(nonempty_schema_frames, ignore_index=True, sort=False)


def resumable_context_frame(
    stage: str,
    shard_root: Path,
    contexts: Iterable[Any],
    context_key: Callable[[Any], Any],
    build_context: Callable[[Any], pd.DataFrame],
    *,
    fingerprint: str,
    string_columns: Optional[Sequence[str]] = None,
    required_columns: Optional[Sequence[str]] = None,
    context_label: Optional[Callable[[Any], str]] = None,
    progress: Optional[ProgressReporter] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a DataFrame from atomic, resumable per-context TSV shards.

    A run directory is derived from ``stage``, the caller's analysis ``fingerprint``,
    and the complete set of context keys. Therefore changing either analysis settings
    or the selected dataset contexts cannot silently reuse an incompatible shard.
    Each completed context is saved before the next starts, so rerunning after an
    exception or kernel interruption reloads completed contexts and resumes the rest.

    Wrap this helper in :func:`cached_frame` when a conventional merged output TSV is
    also desired. ``force_recompute(stage)`` rebuilds every requested context.
    """
    context_list = list(contexts)
    keyed_contexts: list[tuple[Any, Any, str]] = []
    seen_context_hashes: dict[str, Any] = {}
    for context in context_list:
        key = context_key(context)
        key_hash = stable_json_fingerprint(key)
        if key_hash in seen_context_hashes:
            raise ValueError(
                f"[{stage}] duplicate context key: {key!r}; "
                f"first seen as {seen_context_hashes[key_hash]!r}"
            )
        seen_context_hashes[key_hash] = key
        keyed_contexts.append((context, key, key_hash))

    run_fingerprint = stable_json_fingerprint(
        {
            "stage": str(stage),
            "fingerprint": str(fingerprint),
            # Treat context selection as a set so harmless iteration-order changes reuse
            # the same shards; concatenation still follows the caller's current order.
            "context_key_hashes": sorted(seen_context_hashes),
        }
    )
    run_directory = Path(shard_root) / run_fingerprint
    run_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = run_directory / "manifest.json"
    manifest = {
        "stage": str(stage),
        "fingerprint": str(fingerprint),
        "run_fingerprint": run_fingerprint,
        "n_contexts": len(keyed_contexts),
        "context_key_hashes": sorted(seen_context_hashes),
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )

    columns: Sequence[str] = (
        CACHE_STRING_COLUMNS if string_columns is None else tuple(string_columns)
    )
    required = set(required_columns or ())
    force = FORCE_ALL or stage in FORCE_RECOMPUTE
    reporter = progress
    if reporter is None and verbose:
        reporter = ProgressReporter(
            total=len(keyed_contexts),
            label=stage,
            every=max(1, len(keyed_contexts) // 20),
            min_interval_seconds=30.0,
        )

    frames: list[pd.DataFrame] = []
    expected_columns: Optional[list[str]] = None
    n_reloaded = 0
    n_computed = 0
    for context_index, (context, key, key_hash) in enumerate(keyed_contexts, start=1):
        shard_path = run_directory / f"{key_hash}.tsv"
        metadata_path = run_directory / f"{key_hash}.cache.json"
        frame: Optional[pd.DataFrame] = None
        if not force and shard_path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                metadata_columns = metadata.get("columns")
                valid_metadata = (
                    metadata.get("stage") == str(stage)
                    and metadata.get("fingerprint") == str(fingerprint)
                    and metadata.get("run_fingerprint") == run_fingerprint
                    and metadata.get("context_key_hash") == key_hash
                    and isinstance(metadata_columns, list)
                    and required.issubset(metadata_columns)
                )
                if valid_metadata:
                    loaded = _read_tsv(
                        shard_path,
                        string_columns=columns,
                        columns=metadata_columns,
                    )
                    if (
                        loaded.columns.tolist() == metadata_columns
                        and len(loaded) == int(metadata.get("n_rows", -1))
                    ):
                        frame = loaded
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                OSError,
                TypeError,
                ValueError,
                pd.errors.ParserError,
            ):
                frame = None

        if frame is None:
            if verbose:
                if context_label is not None:
                    label = str(context_label(context))
                else:
                    label = json.dumps(
                        _json_ready(key),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                if len(label) > 180:
                    label = f"{label[:177]}..."
                print(
                    f"[{stage}] computing context {context_index:,}/"
                    f"{len(keyed_contexts):,}: {label}",
                    flush=True,
                )
            frame = build_context(context)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"[{stage}] context {key!r} returned "
                    f"{type(frame).__name__}, not a DataFrame"
                )
            missing_columns = required - set(frame.columns)
            if missing_columns:
                raise ValueError(
                    f"[{stage}] context {key!r} omitted required columns: "
                    f"{sorted(missing_columns)!r}"
                )
            metadata = {
                "stage": str(stage),
                "fingerprint": str(fingerprint),
                "run_fingerprint": run_fingerprint,
                "context_key": _json_ready(key),
                "context_key_hash": key_hash,
                "columns": frame.columns.tolist(),
                "n_rows": int(len(frame)),
            }
            _atomic_write_frame(shard_path, frame)
            _atomic_write_text(
                metadata_path,
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            )
            n_computed += 1
        else:
            n_reloaded += 1

        current_columns = frame.columns.tolist()
        if expected_columns is None and current_columns:
            expected_columns = current_columns
        elif current_columns and current_columns != expected_columns:
            raise ValueError(
                f"[{stage}] context {key!r} has columns {current_columns!r}; "
                f"expected {expected_columns!r}"
            )
        frames.append(frame)
        if reporter is not None:
            reporter.update(
                completed=context_index,
                detail=f"{n_reloaded:,} reloaded, {n_computed:,} computed",
            )

    nonempty_schema_frames = [frame for frame in frames if len(frame.columns) > 0]
    if nonempty_schema_frames:
        combined = pd.concat(nonempty_schema_frames, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame(columns=list(required_columns or ()))
    CACHE_STATUS[stage] = (
        "reloaded"
        if n_computed == 0
        else "computed"
        if n_reloaded == 0
        else "resumed"
    )
    if verbose:
        print(
            f"[{stage}] assembled {len(combined):,} rows from "
            f"{n_reloaded:,} cached and {n_computed:,} computed contexts "
            f"in {run_directory.name[:12]}",
            flush=True,
        )
    return combined


def cache_summary() -> pd.DataFrame:
    """One row per stage seen so far, showing whether it was reloaded or recomputed."""
    return pd.DataFrame(
        [{"stage": stage, "status": status} for stage, status in sorted(CACHE_STATUS.items())],
        columns=["stage", "status"],
    )


def _self_test() -> None:
    import tempfile

    import numpy as np

    FORCE_RECOMPUTE.clear()
    CACHE_STATUS.clear()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "stage.tsv"
        frame = pd.DataFrame(
            {
                "left_obs_id": ["00123", "P1_A01"],
                "time_key": ["24", "24"],
                "left_dose_key": ["10", "0.0003"],
                "score": [0.31, np.nan],
                "n_deg": [120, 4],
            }
        )

        calls: list[int] = []
        cached_frame("demo", path, lambda: (calls.append(1), frame)[1], verbose=False)
        assert calls == [1] and CACHE_STATUS["demo"] == "computed"
        assert path.exists(), "parent directories should be created"

        def explode() -> pd.DataFrame:
            raise AssertionError("build() must not run on a cache hit")

        again = cached_frame("demo", path, explode, verbose=False)
        assert CACHE_STATUS["demo"] == "reloaded"
        assert again["left_obs_id"].tolist() == ["00123", "P1_A01"], "leading zeros lost"
        assert again["time_key"].tolist() == ["24", "24"]
        assert again["left_dose_key"].tolist() == ["10", "0.0003"]
        assert again["n_deg"].dtype.kind == "i", again["n_deg"].dtype
        assert np.isclose(again["score"][0], 0.31) and np.isnan(again["score"][1])

        force_recompute("demo")
        calls.clear()
        cached_frame("demo", path, lambda: (calls.append(1), frame)[1], verbose=False)
        assert calls == [1] and CACHE_STATUS["demo"] == "computed"

        force_recompute("other", replace=True)
        assert FORCE_RECOMPUTE == {"other"}
        cached_frame("demo", path, explode, verbose=False)

        fingerprint_path = Path(directory) / "fingerprinted.tsv"
        cached_frame(
            "fingerprinted",
            fingerprint_path,
            lambda: frame,
            fingerprint="v1",
            required_columns=["score"],
            verbose=False,
        )
        fingerprint_calls: list[int] = []
        cached_frame(
            "fingerprinted",
            fingerprint_path,
            lambda: (fingerprint_calls.append(1), frame)[1],
            fingerprint="v2",
            required_columns=["score"],
            verbose=False,
        )
        assert fingerprint_calls == [1], "changed fingerprint must invalidate cache"

        assert is_cached("demo", path) and not is_cached("absent", Path(directory) / "nope.tsv")
        force_recompute("demo")
        assert not is_cached("demo", path), "force_recompute must defeat is_cached"
        force_recompute("other", replace=True)

        summary = cache_summary()
        assert summary["stage"].tolist() == ["demo", "fingerprinted"]
        assert summary.set_index("stage").loc["demo", "status"] == "reloaded"
        assert summary.set_index("stage").loc["fingerprinted", "status"] == "computed"

        try:
            cached_frame("bad", Path(directory) / "bad.tsv", lambda: "not a frame", verbose=False)
        except TypeError as exc:
            assert "not a DataFrame" in str(exc)
        else:
            raise AssertionError("a non-DataFrame build result should raise")

    FORCE_RECOMPUTE.clear()
    CACHE_STATUS.clear()
    print("notebook_cache self-tests passed.")


if __name__ == "__main__":
    _self_test()
