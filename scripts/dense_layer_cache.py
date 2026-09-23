#!/usr/bin/env python3
"""Uncompressed, row-major copies of the DGE layers the replicate scorer reads.

The replicate scorer compares every condition against up to
``--max-baseline-peers`` peers, loaded by row position. The DGE layers are
gzip-compressed in chunks spanning hundreds of rows, so fetching one row
decompresses the whole chunk band it sits in. Measured on Novartis
(55,172 x 20,881, chunks 216 x 164):

====================================  ========  =========
Read                                  Time      Per row
====================================  ========  =========
256 random rows                       38.16 s   149 ms
one contiguous 216-row band            0.26 s     1.2 ms
====================================  ========  =========

Those 256 rows touched 164 of the layer's 256 bands, so each condition
decompressed most of the layer -- twice, for ``logFC`` and ``t`` -- which was
essentially all of the ~68 s per condition. This module decompresses each layer
once, band by band, into a float32 ``.npy`` that every worker memory-maps. The
page cache then holds one shared copy, and a peer row is one contiguous read.

**Results are bit-identical.** Every consumer in the scorer casts layer values
to float32 immediately (``np.asarray(..., dtype=np.float32)``), and casting
commutes with row and column selection, so serving pre-cast values changes
nothing.

Configuration:

``CPB_DENSE_LAYER_CACHE``
    Cache directory, default ``results/dense_layer_cache``; ``off`` disables
    it and every read goes back to the compressed H5AD.
``CPB_DENSE_LAYER_RESERVE_GB``
    Free space to leave on the cache filesystem, default 10. A layer that
    would eat into it is not cached; its reads fall back to the H5AD.

The cache is derived data: deleting it only costs a rebuild.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np

__all__ = ["cache_root", "dense_layers_for", "ensure_dense_layers"]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = REPO_ROOT / "results" / "dense_layer_cache"
DEFAULT_RESERVE_GB = 10.0
CACHE_DTYPE = np.float32
CACHE_FORMAT_VERSION = 1
# Rows decompressed per write; rounded up to whole chunk bands at build time.
TARGET_BAND_BYTES = 256 * 1024 * 1024

_opened: dict[tuple[str, str], np.ndarray] = {}
_opened_lock = threading.Lock()


def cache_root() -> Optional[Path]:
    raw = os.environ.get("CPB_DENSE_LAYER_CACHE", "").strip()
    if raw.lower() in {"off", "0", "false", "no"}:
        return None
    return Path(raw) if raw else DEFAULT_CACHE_ROOT


def _reserve_bytes() -> int:
    raw = os.environ.get("CPB_DENSE_LAYER_RESERVE_GB", "").strip()
    try:
        return int(float(raw) * 1e9) if raw else int(DEFAULT_RESERVE_GB * 1e9)
    except ValueError:
        return int(DEFAULT_RESERVE_GB * 1e9)


def _source_identity(source_path: Path) -> dict[str, object]:
    stat = source_path.stat()
    return {
        "source": str(source_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_paths(root: Path, source_path: Path, layer: str) -> tuple[Path, Path]:
    identity = _source_identity(source_path)
    digest = hashlib.sha256(
        json.dumps(
            {**identity, "layer": layer, "dtype": np.dtype(CACHE_DTYPE).str,
             "version": CACHE_FORMAT_VERSION},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:24]
    stem = f"{source_path.stem}.{layer.replace('/', '_')}.{digest}"
    return root / f"{stem}.npy", root / f"{stem}.json"


def _expected_metadata(source_path: Path, layer: str, shape: Sequence[int]) -> dict:
    return {
        **_source_identity(source_path),
        "layer": layer,
        "shape": [int(value) for value in shape],
        "dtype": np.dtype(CACHE_DTYPE).str,
        "version": CACHE_FORMAT_VERSION,
    }


def _load_if_valid(npy_path: Path, meta_path: Path, expected: dict) -> Optional[np.ndarray]:
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata != expected:
        return None
    try:
        array = np.load(npy_path, mmap_mode="r")
    except (OSError, ValueError):
        return None
    if list(array.shape) != expected["shape"] or array.dtype != np.dtype(CACHE_DTYPE):
        return None
    return array


class _FileLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: Optional[int] = None

    def __enter__(self) -> "_FileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


def _build_layer(
    handle: h5py.File,
    layer: str,
    npy_path: Path,
    meta_path: Path,
    expected: dict,
    verbose: bool,
) -> None:
    dataset = handle["layers"][layer]
    n_rows, n_cols = dataset.shape
    chunk_rows = int(dataset.chunks[0]) if dataset.chunks else 1024
    bytes_per_row = max(1, n_cols * dataset.dtype.itemsize)
    bands = max(1, TARGET_BAND_BYTES // (bytes_per_row * chunk_rows))
    step = chunk_rows * bands

    temporary = npy_path.with_name(f".{npy_path.name}.tmp-{os.getpid()}")
    started = time.monotonic()
    try:
        out = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=CACHE_DTYPE, shape=(n_rows, n_cols)
        )
        # Whole chunk bands, in order, so each gzip chunk is decompressed once.
        for start in range(0, n_rows, step):
            stop = min(start + step, n_rows)
            out[start:stop] = np.asarray(dataset[start:stop, :], dtype=CACHE_DTYPE)
        out.flush()
        del out
        with open(temporary, "rb+") as flushed:
            os.fsync(flushed.fileno())
        os.replace(temporary, npy_path)
        meta_temporary = meta_path.with_name(f".{meta_path.name}.tmp-{os.getpid()}")
        meta_temporary.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
        os.replace(meta_temporary, meta_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if verbose:
        elapsed = time.monotonic() - started
        size_gb = n_rows * n_cols * np.dtype(CACHE_DTYPE).itemsize / 1e9
        print(
            f"[dense-cache] built {npy_path.name}: {n_rows:,} x {n_cols:,} "
            f"({size_gb:.1f} GB) in {elapsed:.0f}s",
            flush=True,
        )


def ensure_dense_layers(
    source_path: str | os.PathLike[str],
    layers: Iterable[str],
    *,
    prewarm_layers: Sequence[str] = (),
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Return memory-mapped float32 copies of ``layers``, building any missing.

    Layers that cannot be cached -- disabled, absent, or no disk room -- are
    simply left out of the result, and the caller reads them from the H5AD
    as before. Concurrent workers serialize on a per-source lock, so each
    layer is built exactly once.

    ``prewarm_layers`` should be every layer the caller will read from this
    source, not just the ones being built: page-cache warm records are kept
    per file, so warming a subset here would make a later warm of the rest
    look already done.
    """
    root = cache_root()
    if root is None:
        return {}
    source_path = Path(source_path)
    wanted = [str(layer) for layer in layers]
    result: dict[str, np.ndarray] = {}

    with _opened_lock:
        for layer in wanted:
            key = (str(source_path.resolve()), layer)
            if key in _opened:
                result[layer] = _opened[key]
    missing = [layer for layer in wanted if layer not in result]
    if not missing:
        return result

    lock_path = root / f".{source_path.stem}.{hashlib.sha256(str(source_path.resolve()).encode()).hexdigest()[:16]}.lock"
    try:
        with _FileLock(lock_path):
            with h5py.File(source_path, "r") as handle:
                layer_group = handle.get("layers")
                to_build: list[tuple[str, Path, Path, dict]] = []
                for layer in missing:
                    if layer_group is None or layer not in layer_group:
                        continue
                    shape = layer_group[layer].shape
                    npy_path, meta_path = _cache_paths(root, source_path, layer)
                    expected = _expected_metadata(source_path, layer, shape)
                    cached = _load_if_valid(npy_path, meta_path, expected)
                    if cached is not None:
                        result[layer] = cached
                    else:
                        to_build.append((layer, npy_path, meta_path, expected))

                if to_build:
                    needed = sum(
                        int(np.prod(expected["shape"])) * np.dtype(CACHE_DTYPE).itemsize
                        for _, _, _, expected in to_build
                    )
                    root.mkdir(parents=True, exist_ok=True)
                    free = shutil.disk_usage(root).free
                    if free - needed < _reserve_bytes():
                        if verbose:
                            print(
                                f"[dense-cache] {source_path.name}: {needed / 1e9:.1f} GB "
                                f"would leave {(free - needed) / 1e9:.1f} GB free, under "
                                f"the {_reserve_bytes() / 1e9:.0f} GB reserve; reading "
                                "from the H5AD instead",
                                flush=True,
                            )
                        to_build = []
                if to_build:
                    # Band reads are sequential; warm the source first so they
                    # come from RAM rather than 128 KB readahead.
                    try:
                        from page_cache import prewarm_h5ad
                    except ImportError:
                        from scripts.page_cache import prewarm_h5ad
                    prewarm_h5ad(
                        source_path,
                        layer_names=(),
                        optional_layer_names=tuple(prewarm_layers) or tuple(wanted),
                        verbose=verbose,
                    )
                    for layer, npy_path, meta_path, expected in to_build:
                        _build_layer(handle, layer, npy_path, meta_path, expected, verbose)
                        cached = _load_if_valid(npy_path, meta_path, expected)
                        if cached is not None:
                            result[layer] = cached
    except OSError as exc:
        if exc.errno not in (errno.ENOSPC, errno.EDQUOT) and verbose:
            print(f"[dense-cache] {source_path.name}: {exc}; reading from the H5AD", flush=True)
        elif verbose:
            print(f"[dense-cache] {source_path.name}: out of space; reading from the H5AD", flush=True)

    with _opened_lock:
        for layer, array in result.items():
            _opened[(str(source_path.resolve()), layer)] = array
    return result


def dense_layers_for(source_path: str | os.PathLike[str]) -> Mapping[str, np.ndarray]:
    """Layers already opened for ``source_path`` in this process."""
    resolved = str(Path(source_path).resolve())
    with _opened_lock:
        return {layer: array for (path, layer), array in _opened.items() if path == resolved}


def _self_test() -> None:
    import tempfile

    import anndata as ad
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        os.environ["CPB_DENSE_LAYER_CACHE"] = str(tmp_path / "cache")
        os.environ["CPB_DENSE_LAYER_RESERVE_GB"] = "0"
        os.environ["CPB_PAGE_CACHE_PREWARM_LOCK_TIMEOUT_S"] = "5"
        rng = np.random.default_rng(11)
        rows, cols = 700, 90
        logfc = rng.normal(size=(rows, cols))
        logfc[3, 7] = np.nan
        t_stat = rng.normal(size=(rows, cols)) * 5
        adata = ad.AnnData(
            X=np.zeros((rows, cols), dtype=np.float32),
            obs=pd.DataFrame(index=[f"r{i}" for i in range(rows)]),
            var=pd.DataFrame(index=[f"g{j}" for j in range(cols)]),
        )
        adata.layers["logFC"] = logfc
        adata.layers["t"] = t_stat
        source = tmp_path / "source_de.h5ad"
        adata.write_h5ad(source, compression="gzip")
        with h5py.File(source, "r+") as handle:
            # Re-chunk like the real files: tall bands, so band boundaries matter.
            data = handle["layers"]["logFC"][...]
            del handle["layers"]["logFC"]
            handle["layers"].create_dataset(
                "logFC", data=data, chunks=(64, 30), compression="gzip"
            )

        built = ensure_dense_layers(source, ["logFC", "t", "absent"], verbose=False)
        assert set(built) == {"logFC", "t"}, built.keys()

        positions = np.array([699, 3, 3, 0, 65, 64, 128], dtype=np.int64)
        genes = np.array([7, 0, 89, 7], dtype=np.int64)
        with h5py.File(source, "r") as handle:
            for layer in ("logFC", "t"):
                reference = np.asarray(
                    handle["layers"][layer][...][positions][:, genes], dtype=np.float32
                )
                served = np.asarray(built[layer][positions][:, genes], dtype=np.float32)
                # Bit-identical, NaN included.
                assert np.array_equal(reference, served, equal_nan=True), layer
                full = np.asarray(handle["layers"][layer][...], dtype=np.float32)
                assert np.array_equal(full, np.asarray(built[layer]), equal_nan=True)

        # A second request is served from this process's memo.
        again = ensure_dense_layers(source, ["logFC"], verbose=False)
        assert again["logFC"] is built["logFC"]
        assert set(dense_layers_for(source)) == {"logFC", "t"}

        # A fresh process would reload from disk rather than rebuild.
        with _opened_lock:
            _opened.clear()
        reloaded = ensure_dense_layers(source, ["logFC"], verbose=False)
        assert np.array_equal(
            np.asarray(reloaded["logFC"]), np.asarray(built["logFC"]), equal_nan=True
        )

        # Changing the source invalidates the cache.
        with _opened_lock:
            _opened.clear()
        os.utime(source, ns=(time.time_ns(), time.time_ns() + 10**9))
        npy_before = sorted((tmp_path / "cache").glob("*.npy"))
        ensure_dense_layers(source, ["logFC"], verbose=False)
        npy_after = sorted((tmp_path / "cache").glob("*.npy"))
        assert len(npy_after) == len(npy_before) + 1, (npy_before, npy_after)

        # The reserve is honoured: nothing is cached, and nothing breaks.
        with _opened_lock:
            _opened.clear()
        os.environ["CPB_DENSE_LAYER_RESERVE_GB"] = "1e9"
        other = tmp_path / "other_de.h5ad"
        shutil.copy(source, other)
        assert ensure_dense_layers(other, ["logFC"], verbose=False) == {}

        os.environ["CPB_DENSE_LAYER_CACHE"] = "off"
        assert ensure_dense_layers(source, ["logFC"], verbose=False) == {}
        for name in (
            "CPB_DENSE_LAYER_CACHE",
            "CPB_DENSE_LAYER_RESERVE_GB",
            "CPB_PAGE_CACHE_PREWARM_LOCK_TIMEOUT_S",
        ):
            os.environ.pop(name, None)
    print("dense_layer_cache self-tests passed")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _self_test()
