#!/usr/bin/env python3
"""Open a DGE H5AD without pulling every layer into memory.

``anndata``'s backed mode only backs ``X``.  ``read_h5ad_backed`` does::

    attributes = ["obsm", "varm", "obsp", "varp", "uns", "layers"]
    d.update({k: read_elem(f[k]) for k in attributes if k in f})

so ``layers`` is read eagerly.  The DGE files in this project have **no**
``X`` — only eleven same-shaped ``float64`` layers — which makes a backed
open the most expensive way to read them.  Measured on this box:

===================================  =========  ========
File                                 backed     lazy
===================================  =========  ========
op3 group_rep (54 MB)                   12.3 s     0.0 s
vcpi_0002 group_rep (9.6 GB)          1204.4 s  seconds
===================================  =========  ========

The backed open of vcpi_0002 also held 13.8 GB resident and handed back
``layers["logFC"]`` as a plain ``ndarray``.  On the old 30 GB box, Novartis
(~23 GB of layers) could not be opened at all.

:class:`LazyH5AD` reads ``obs``/``var`` exactly the way ``anndata`` does and
leaves each layer as an ``h5py`` dataset, so callers touch only the layers
they index.  It exposes the attribute surface the scoring code already
uses — ``obs``, ``var``, ``var_names``, ``obs_names``, ``shape``, ``n_obs``,
``n_vars``, ``layers``, ``file``, ``close()`` — and layer indexing accepts
the same keys a NumPy array would, which plain ``h5py`` does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

__all__ = ["LazyH5AD", "LazyLayer", "open_lazy_h5ad"]


def _read_dataframe(handle: h5py.File, key: str) -> pd.DataFrame:
    """Read ``obs``/``var`` the same way ``anndata`` would for this file."""
    from anndata._io.specs import read_elem

    node = handle.get(key)
    if node is None:
        return pd.DataFrame()
    if "encoding-type" in handle.attrs or "encoding-type" in node.attrs:
        frame = read_elem(node)
    else:  # pre-0.7 layout, matching anndata's own backwards-compat branch
        from anndata._io.h5ad import read_dataframe

        frame = read_dataframe(node)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{key} did not decode to a DataFrame: {type(frame)!r}")
    return frame


def _is_index_array(key: Any) -> bool:
    """True for an integer fancy-index that h5py would reject as-is."""
    if isinstance(key, np.ndarray):
        return key.ndim == 1 and np.issubdtype(key.dtype, np.integer)
    if isinstance(key, (list, tuple)) and len(key) > 0:
        return all(
            isinstance(item, (int, np.integer)) and not isinstance(item, bool)
            for item in key
        )
    return False


class LazyLayer:
    """One on-disk layer, indexable like the in-memory array it replaces.

    ``h5py`` only accepts a strictly increasing fancy index with no repeats,
    while the scoring code passes row positions in whatever order the rows
    came in and sometimes repeats them.  Rows are therefore fetched once in
    sorted order and expanded back to the caller's order, which keeps results
    identical to indexing a NumPy array while still reading each row once.
    """

    __slots__ = ("_dataset",)

    def __init__(self, dataset: h5py.Dataset) -> None:
        self._dataset = dataset

    @property
    def dataset(self) -> h5py.Dataset:
        return self._dataset

    @property
    def shape(self) -> tuple[int, ...]:
        return self._dataset.shape

    @property
    def dtype(self) -> np.dtype:
        return self._dataset.dtype

    @property
    def ndim(self) -> int:
        return self._dataset.ndim

    def __len__(self) -> int:
        return self._dataset.shape[0]

    def _read_rows(self, rows: Any) -> np.ndarray:
        positions = np.asarray(rows, dtype=np.int64)
        if positions.size == 0:
            return np.empty((0,) + tuple(self._dataset.shape[1:]), self._dataset.dtype)
        unique, inverse = np.unique(positions, return_inverse=True)
        block = self._dataset[unique.tolist(), ...]
        block = np.asarray(block)
        if block.ndim < self._dataset.ndim:
            # A one-element selection can come back with the axis dropped.
            block = block[np.newaxis, ...]
        return block[inverse]

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            if not key:
                return np.asarray(self._dataset[...])
            first, rest = key[0], key[1:]
            if _is_index_array(first):
                block = self._read_rows(first)
                return block[(slice(None),) + rest] if rest else block
            return np.asarray(self._dataset[key])
        if _is_index_array(key):
            return self._read_rows(key)
        return np.asarray(self._dataset[key])

    def __array__(self, dtype: Any = None) -> np.ndarray:
        values = np.asarray(self._dataset[...])
        return values.astype(dtype) if dtype is not None else values

    def __repr__(self) -> str:
        return f"LazyLayer(shape={self.shape}, dtype={self.dtype}, on disk)"


class LazyLayers(Mapping):
    """Mapping of layer name to :class:`LazyLayer`, resolved on access."""

    def __init__(self, group: h5py.Group | None) -> None:
        self._group = group
        self._names: tuple[str, ...] = (
            tuple(group.keys()) if group is not None else ()
        )
        self._cache: dict[str, LazyLayer] = {}

    def __getitem__(self, name: str) -> LazyLayer:
        if name not in self._cache:
            if self._group is None or name not in self._group:
                raise KeyError(name)
            node = self._group[name]
            if not isinstance(node, h5py.Dataset):
                raise TypeError(f"layer {name!r} is not a dataset")
            self._cache[name] = LazyLayer(node)
        return self._cache[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def keys(self):  # noqa: D102 - Mapping.keys, spelled out for clarity
        return self._names

    def __repr__(self) -> str:
        return f"LazyLayers({list(self._names)!r})"


class LazyH5AD:
    """A read-only, layer-lazy stand-in for a backed ``AnnData``."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._handle = h5py.File(self._path, "r")
        try:
            self._obs = _read_dataframe(self._handle, "obs")
            self._var = _read_dataframe(self._handle, "var")
            self._layers = LazyLayers(self._handle.get("layers"))
        except Exception:
            self._handle.close()
            raise

    @property
    def path(self) -> Path:
        return self._path

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @property
    def var(self) -> pd.DataFrame:
        return self._var

    @property
    def obs_names(self) -> pd.Index:
        return self._obs.index

    @property
    def var_names(self) -> pd.Index:
        return self._var.index

    @property
    def n_obs(self) -> int:
        return int(len(self._obs))

    @property
    def n_vars(self) -> int:
        return int(len(self._var))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_obs, self.n_vars)

    @property
    def layers(self) -> LazyLayers:
        return self._layers

    @property
    def file(self) -> h5py.File | None:
        """The open handle, so ``adata.file.close()`` keeps working."""
        return self._handle if self._handle else None

    @property
    def isbacked(self) -> bool:
        return True

    def close(self) -> None:
        if self._handle:
            self._handle.close()

    def __enter__(self) -> "LazyH5AD":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"LazyH5AD({self._path.name}, {self.n_obs} x {self.n_vars}, "
            f"layers={list(self._layers.keys())!r})"
        )


def open_lazy_h5ad(path: str | Path) -> LazyH5AD:
    """Open ``path`` reading only ``obs``/``var`` eagerly."""
    return LazyH5AD(path)


def _self_test() -> None:
    import tempfile

    import anndata as ad

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.h5ad"
        rows, cols = 40, 25
        rng = np.random.default_rng(7)
        payload = {
            name: rng.normal(size=(rows, cols))
            for name in ("logFC", "t", "adj.P.Value.within_one_contrast")
        }
        obs = pd.DataFrame(
            {
                "cell_type": ["CVCL_0001"] * rows,
                "pert_dose_uM": rng.uniform(size=rows),
                "is_control": [False] * rows,
            },
            index=[f"cond{i}" for i in range(rows)],
        )
        var = pd.DataFrame(
            {"symbol": [f"GENE{j}" for j in range(cols)]},
            index=[f"ENSG{j:05d}" for j in range(cols)],
        )
        written = ad.AnnData(
            X=np.zeros((rows, cols), dtype=np.float32), obs=obs, var=var
        )
        for name, values in payload.items():
            written.layers[name] = values
        written.write_h5ad(path)

        reference = ad.read_h5ad(path)
        with open_lazy_h5ad(path) as lazy:
            assert lazy.shape == reference.shape, (lazy.shape, reference.shape)
            assert lazy.n_obs == reference.n_obs
            assert list(lazy.layers.keys()) == list(reference.layers.keys())
            pd.testing.assert_frame_equal(lazy.obs, reference.obs)
            pd.testing.assert_frame_equal(lazy.var, reference.var)
            assert list(lazy.var_names) == list(reference.var_names)

            layer = lazy.layers["logFC"]
            ref = np.asarray(reference.layers["logFC"])

            # Row-block slicing, as the population fitter uses.
            np.testing.assert_array_equal(layer[0:16, :], ref[0:16, :])
            np.testing.assert_array_equal(layer[:], ref[:])

            # Unsorted, repeated fancy rows, as the replicate scorer uses.
            positions = np.array([9, 2, 9, 31, 0, 2], dtype=np.int64)
            genes = np.array([4, 1, 20], dtype=np.int64)
            np.testing.assert_array_equal(
                layer[positions][:, genes], ref[positions][:, genes]
            )
            np.testing.assert_array_equal(layer[positions], ref[positions])

            # Single row must keep its leading axis, so callers that iterate
            # rows behave the same as before.
            one = np.array([5], dtype=np.int64)
            assert layer[one].shape == ref[one].shape, layer[one].shape
            np.testing.assert_array_equal(layer[one], ref[one])

            # Empty selection.
            empty = np.array([], dtype=np.int64)
            assert layer[empty].shape == ref[empty].shape

            np.testing.assert_array_equal(np.asarray(layer), ref)

            try:
                lazy.layers["absent"]
            except KeyError:
                pass
            else:
                raise AssertionError("expected KeyError for a missing layer")

        # A file with no X at all, which is the real shape of the DGE outputs.
        no_x = Path(tmp) / "no_x.h5ad"
        with h5py.File(path, "r") as src, h5py.File(no_x, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for key in src:
                if key != "X":
                    src.copy(key, dst, name=key)
        with open_lazy_h5ad(no_x) as lazy:
            assert lazy.shape == (rows, cols), lazy.shape
            np.testing.assert_array_equal(
                np.asarray(lazy.layers["t"]), np.asarray(reference.layers["t"])
            )
            assert isinstance(lazy.layers["t"].dataset, h5py.Dataset)

    print("lazy_h5ad self-tests passed")


if __name__ == "__main__":
    _self_test()
