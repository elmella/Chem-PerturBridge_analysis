#!/usr/bin/env python3
"""Page-cache pre-warming for large chunked H5AD reads.

The DGE H5ADs store each layer as gzip-compressed ``(142, 113)`` chunks, so a
row-block read scatters hundreds of ~100 KB requests across the file.  This
volume caps ``read_ahead_kb`` at 128 and roughly fixes the cost of a request
regardless of its size, so a single buffered stream is latency-bound no matter
how the bytes are laid out:

============================  ==========
Access pattern                Throughput
============================  ==========
1 buffered stream              ~4.3 MB/s
8 buffered streams            ~16.5 MB/s
64 buffered streams           ~24.8 MB/s
``dd bs=4M iflag=direct``      ~155 MB/s
============================  ==========

A plain end-to-end pre-read therefore buys nothing: it is exactly as slow as
the access pattern it was meant to fix.  Two things make the pre-pass pay:
issuing many requests concurrently, and warming only the layers the caller
will read (chunks of one layer are stored contiguously, so a layer is a single
byte range).  Warming just ``logFC`` on a 9.6 GB source reads 1.1 GB at
~23 MB/s.

This is a pure read-ahead optimisation: it never changes what is computed, and
disabling it only costs time.

Configuration (all optional):

``CPB_PAGE_CACHE_PREWARM``
    ``blocking`` (default), ``background``, or ``off``.  ``background`` starts
    the scan on a daemon thread and returns immediately, letting the reader
    race a warmer that is far faster than it is.
``CPB_PAGE_CACHE_PREWARM_BUDGET_GB``
    Cap on bytes warmed per file.  Defaults to 70% of ``MemAvailable`` at call
    time, so a file larger than RAM warms its head rather than thrashing.
``CPB_PAGE_CACHE_PREWARM_BLOCK_MB``
    Per-request block size, default 8.
``CPB_PAGE_CACHE_PREWARM_THREADS``
    Concurrent read streams.  Defaults to ``4x`` the core count, capped at
    128, because these streams block on I/O rather than CPU.
``CPB_PAGE_CACHE_PREWARM_LOCK_TIMEOUT_S``
    How long to wait for another process warming the same file, default 1800.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import itertools
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Iterable, Sequence

__all__ = [
    "PrewarmResult",
    "prewarm_file",
    "prewarm_files",
    "prewarm_h5ad",
    "prewarm_mode",
]


MODE_BLOCKING = "blocking"
MODE_BACKGROUND = "background"
MODE_OFF = "off"
_VALID_MODES = (MODE_BLOCKING, MODE_BACKGROUND, MODE_OFF)

DEFAULT_BLOCK_MB = 8


def _default_threads() -> int:
    """Concurrent read streams to use when nothing is configured.

    These streams block on I/O rather than CPU, so the useful count is well
    above the core count; measured throughput rose from 4.3 MB/s at one
    stream to ~25 MB/s at 64 and ~29 MB/s at 96. Scaling off the core count
    keeps a small box from oversubscribing while letting a large one reach
    the plateau.
    """
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 4
    return int(min(128, max(16, cores * 4)))


DEFAULT_BUDGET_FRACTION = 0.70
_MIN_BUDGET_BYTES = 256 * 1024 * 1024

# Paths this process already warmed. Page cache is shared, but re-reading a
# multi-GB file because two scans touch it in sequence would undo the saving.
_warmed_paths: set[tuple[str, int, int]] = set()
_warmed_lock = threading.Lock()


@dataclass(frozen=True)
class PrewarmResult:
    """Outcome of one pre-warm attempt."""

    path: Path
    mode: str
    file_bytes: int
    warmed_bytes: int
    elapsed_seconds: float
    skipped_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.skipped_reason is None and self.warmed_bytes >= self.file_bytes

    @property
    def throughput_mb_s(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.warmed_bytes / 1e6 / self.elapsed_seconds

    def describe(self) -> str:
        if self.skipped_reason is not None:
            return f"skipped ({self.skipped_reason})"
        scope = "full" if self.complete else "partial"
        return (
            f"{scope} {self.warmed_bytes / 1e9:.1f}/{self.file_bytes / 1e9:.1f} GB "
            f"in {self.elapsed_seconds:.1f}s ({self.throughput_mb_s:.0f} MB/s)"
        )


def prewarm_mode() -> str:
    """Resolve the configured mode, falling back to blocking on a bad value."""
    raw = os.environ.get("CPB_PAGE_CACHE_PREWARM", "").strip().lower()
    if not raw:
        return MODE_BLOCKING
    if raw in _VALID_MODES:
        return raw
    # Accept the obvious boolean spellings so a 0/1 toggle behaves sanely.
    if raw in {"0", "false", "no"}:
        return MODE_OFF
    if raw in {"1", "true", "yes"}:
        return MODE_BLOCKING
    return MODE_BLOCKING


def _available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _budget_bytes() -> int:
    raw = os.environ.get("CPB_PAGE_CACHE_PREWARM_BUDGET_GB", "").strip()
    if raw:
        try:
            return max(_MIN_BUDGET_BYTES, int(float(raw) * 1e9))
        except ValueError:
            pass
    available = _available_memory_bytes()
    if available is None:
        # Without a reading, warm a conservative fixed slice rather than
        # guessing high and evicting the caller's own working set.
        return 4 * 1024 * 1024 * 1024
    return max(_MIN_BUDGET_BYTES, int(available * DEFAULT_BUDGET_FRACTION))


def _lock_timeout_seconds() -> float:
    raw = os.environ.get("CPB_PAGE_CACHE_PREWARM_LOCK_TIMEOUT_S", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return DEFAULT_LOCK_TIMEOUT_SECONDS


def _block_size() -> int:
    raw = os.environ.get("CPB_PAGE_CACHE_PREWARM_BLOCK_MB", "").strip()
    if raw:
        try:
            return max(1024 * 1024, int(float(raw) * 1024 * 1024))
        except ValueError:
            pass
    return DEFAULT_BLOCK_MB * 1024 * 1024


def _thread_count() -> int:
    raw = os.environ.get("CPB_PAGE_CACHE_PREWARM_THREADS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return _default_threads()


def _file_identity(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _boot_id() -> str:
    """Identify this boot, so warm-cache records expire with the page cache."""
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as handle:
            return handle.read().strip().replace(":", "-")
    except OSError:
        return "unknown-boot"


LOCK_POLL_SECONDS = 2.0
DEFAULT_LOCK_TIMEOUT_SECONDS = 30 * 60


class _PrewarmLock:
    """Cross-process guard so N workers warm a shared file once, not N times.

    Workers of the replicate scorer routinely open the same source H5AD, and
    they must not race each other: concurrent readers of a cold file contend
    for exactly the request budget the warm-up is trying to spend
    sequentially.  So a caller that loses the race waits for the winner and
    then reads from RAM, rather than starting its own cold scan.

    The holder records ``boot_id:size:mtime_ns:warmed_bytes`` in the lock file, which
    lets a waiter recognise that the work it was waiting for is already done.
    """

    def __init__(self, key: str, timeout_seconds: float) -> None:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        self._path = Path(tempfile.gettempdir()) / f"cpb_prewarm_{digest}.lock"
        self._timeout_seconds = timeout_seconds
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Block until held, or return False once the timeout elapses."""
        try:
            import fcntl
        except ImportError:
            # No usable lock: warming twice is wasteful but harmless.
            return True

        deadline = time.monotonic() + self._timeout_seconds
        while True:
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
            except OSError:
                return True
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(fd)
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    return True
                if time.monotonic() >= deadline:
                    return False
                time.sleep(LOCK_POLL_SECONDS)
                continue
            self._fd = fd
            return True

    def completed_bytes(self, identity: tuple[str, int, int]) -> int:
        """Bytes a previous holder warmed for this exact file revision."""
        if self._fd is None:
            return 0
        try:
            os.lseek(self._fd, 0, os.SEEK_SET)
            raw = os.read(self._fd, 256).decode("utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return 0
        parts = raw.split(":")
        if len(parts) != 4:
            return 0
        boot, *numbers = parts
        # The page cache does not survive a reboot even where /tmp does, so a
        # record from an earlier boot says nothing about what is in RAM now.
        if boot != _boot_id():
            return 0
        try:
            size, mtime_ns, warmed = (int(part) for part in numbers)
        except ValueError:
            return 0
        if (size, mtime_ns) != (identity[1], identity[2]):
            return 0
        return warmed

    def record(self, identity: tuple[str, int, int], warmed_bytes: int) -> None:
        if self._fd is None:
            return
        payload = (
            f"{_boot_id()}:{identity[1]}:{identity[2]}:{int(warmed_bytes)}"
        ).encode("utf-8")
        try:
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.ftruncate(self._fd, 0)
            os.write(self._fd, payload)
        except OSError:
            pass

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        finally:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


def _read_range(path: Path, offset: int, length: int, block_size: int) -> int:
    """Pull one byte range into the page cache and report bytes read."""
    read_bytes = 0
    buffer = bytearray(min(block_size, max(length, 1)))
    view = memoryview(buffer)
    with open(path, "rb", buffering=0) as handle:
        handle.seek(offset)
        while read_bytes < length:
            want = min(len(view), length - read_bytes)
            got = handle.readinto(view[:want])
            if not got:
                break
            read_bytes += got
    return read_bytes


def _split_ranges(
    ranges: Sequence[tuple[int, int]], piece_bytes: int
) -> list[tuple[int, int]]:
    """Cut ranges into work items no larger than ``piece_bytes``."""
    pieces: list[tuple[int, int]] = []
    for offset, length in ranges:
        position = offset
        remaining = length
        while remaining > 0:
            take = min(piece_bytes, remaining)
            pieces.append((position, take))
            position += take
            remaining -= take
    return pieces


def _parallel_read(
    path: Path,
    ranges: Sequence[tuple[int, int]],
    block_size: int,
    threads: int,
) -> int:
    """Read ranges concurrently.

    Readahead on this storage is capped at 128 KB per stream, so a single
    sequential reader is latency-bound at a small fraction of the volume's
    throughput.  Issuing many requests at once is the only lever available
    without root, and it is worth roughly a 6x speed-up.
    """
    pieces = _split_ranges(ranges, block_size)
    if not pieces:
        return 0
    if threads <= 1 or len(pieces) == 1:
        return sum(_read_range(path, off, length, block_size) for off, length in pieces)

    counts = [0] * len(pieces)
    next_index = itertools.count()
    lock = threading.Lock()

    def worker() -> None:
        while True:
            with lock:
                index = next(next_index)
            if index >= len(pieces):
                return
            offset, length = pieces[index]
            try:
                counts[index] = _read_range(path, offset, length, block_size)
            except OSError:
                counts[index] = 0

    pool = [
        threading.Thread(target=worker, name=f"prewarm-read-{i}", daemon=True)
        for i in range(min(threads, len(pieces)))
    ]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    return sum(counts)


def _sequential_read(path: Path, limit_bytes: int, block_size: int) -> int:
    """Warm the first ``limit_bytes`` of a file."""
    return _parallel_read(path, [(0, limit_bytes)], block_size, _thread_count())


def prewarm_file(
    path: str | os.PathLike[str],
    *,
    label: str | None = None,
    mode: str | None = None,
    budget_bytes: int | None = None,
    verbose: bool = True,
) -> PrewarmResult:
    """Pull ``path`` into the page cache with one sequential scan.

    Returns immediately in ``background`` mode; the reported result then
    describes the launch rather than a completed scan.
    """
    path = Path(path)
    mode = (mode or prewarm_mode()).lower()
    display = label or path.name

    if mode == MODE_OFF:
        return PrewarmResult(path, mode, 0, 0, 0.0, "disabled")

    identity = _file_identity(path)
    if identity is None:
        return PrewarmResult(path, mode, 0, 0, 0.0, "unreadable")
    file_bytes = identity[1]
    if file_bytes <= 0:
        return PrewarmResult(path, mode, file_bytes, 0, 0.0, "empty")

    with _warmed_lock:
        if identity in _warmed_paths:
            return PrewarmResult(path, mode, file_bytes, 0, 0.0, "already warmed")
        _warmed_paths.add(identity)

    budget = budget_bytes if budget_bytes is not None else _budget_bytes()
    limit = min(file_bytes, budget)
    block_size = _block_size()

    def _run() -> PrewarmResult:
        started = time.monotonic()
        lock = _PrewarmLock(str(path), _lock_timeout_seconds())
        if not lock.acquire():
            return PrewarmResult(
                path,
                mode,
                file_bytes,
                0,
                time.monotonic() - started,
                "timed out waiting for another process to warm it",
            )
        try:
            already = lock.completed_bytes(identity)
            if already >= limit:
                result = PrewarmResult(
                    path,
                    mode,
                    file_bytes,
                    already,
                    time.monotonic() - started,
                    "warmed by another process",
                )
                if verbose:
                    print(f"[prewarm] {display}: {result.describe()}", flush=True)
                return result
            try:
                read_bytes = _sequential_read(path, limit, block_size)
            except OSError as exc:
                # Let a later caller retry rather than recording a warm that
                # never happened.
                with _warmed_lock:
                    _warmed_paths.discard(identity)
                return PrewarmResult(
                    path, mode, file_bytes, 0, time.monotonic() - started, str(exc)
                )
            lock.record(identity, read_bytes)
        finally:
            lock.release()
        result = PrewarmResult(
            path, mode, file_bytes, read_bytes, time.monotonic() - started
        )
        if verbose:
            print(f"[prewarm] {display}: {result.describe()}", flush=True)
        return result

    if mode == MODE_BACKGROUND:
        if verbose:
            print(
                f"[prewarm] {display}: streaming {limit / 1e9:.1f} GB "
                "in the background",
                flush=True,
            )
        thread = threading.Thread(
            target=_run, name=f"prewarm-{display}", daemon=True
        )
        thread.start()
        return PrewarmResult(path, mode, file_bytes, 0, 0.0, "started in background")

    if verbose and limit < file_bytes:
        print(
            f"[prewarm] {display}: {file_bytes / 1e9:.1f} GB exceeds the "
            f"{budget / 1e9:.1f} GB cache budget; warming the first "
            f"{limit / 1e9:.1f} GB",
            flush=True,
        )
    return _run()


def _dataset_byte_ranges(dataset) -> list[tuple[int, int]]:  # type: ignore[no-untyped-def]
    """Byte ranges backing one HDF5 dataset, or an empty list if unknown.

    Chunked datasets written in one pass store their chunks contiguously, so
    the common case collapses to a single range derived from the first and
    last chunk.  Enumerating all chunks would itself be a slow metadata scan,
    so contiguity is verified on a sample and anything irregular falls back to
    the caller's whole-file path.
    """
    dsid = dataset.id
    if dataset.chunks is None:
        try:
            offset = dsid.get_offset()
        except (AttributeError, ValueError):
            return []
        if offset is None:
            return []
        return [(int(offset), int(dsid.get_storage_size()))]

    try:
        n_chunks = dsid.get_num_chunks()
    except (AttributeError, ValueError, OSError):
        return []
    if n_chunks <= 0:
        return []

    try:
        first = dsid.get_chunk_info(0)
        last = dsid.get_chunk_info(n_chunks - 1)
    except (AttributeError, ValueError, OSError):
        return []

    start = int(first.byte_offset)
    end = int(last.byte_offset) + int(last.size)
    span = end - start
    if span <= 0:
        return []

    storage = int(dsid.get_storage_size())
    # A span far larger than the stored bytes means the chunks are scattered,
    # and warming the gap would read unrelated parts of the file.
    if storage > 0 and span > storage * 2:
        return []

    sample_count = min(8, n_chunks)
    try:
        for index in range(sample_count):
            probe = dsid.get_chunk_info(index * (n_chunks - 1) // max(sample_count - 1, 1))
            if not start <= int(probe.byte_offset) < end:
                return []
    except (AttributeError, ValueError, OSError):
        return []

    return [(start, span)]


def _merge_ranges(
    ranges: Iterable[tuple[int, int]], gap_bytes: int = 4 * 1024 * 1024
) -> list[tuple[int, int]]:
    """Coalesce overlapping or near-adjacent ranges to cut request count."""
    ordered = sorted((int(o), int(n)) for o, n in ranges if n > 0)
    merged: list[tuple[int, int]] = []
    for offset, length in ordered:
        if merged and offset <= merged[-1][0] + merged[-1][1] + gap_bytes:
            prev_offset, prev_length = merged[-1]
            new_end = max(prev_offset + prev_length, offset + length)
            merged[-1] = (prev_offset, new_end - prev_offset)
        else:
            merged.append((offset, length))
    return merged


def _h5ad_prewarm_ranges(
    path: Path,
    layer_names: Sequence[str],
    optional_layer_names: Sequence[str] = (),
) -> list[tuple[int, int]] | None:
    """Byte ranges covering the requested layers plus obs/var metadata.

    Returns ``None`` when the layout cannot be resolved, which tells the
    caller to fall back to warming the whole file.
    """
    try:
        import h5py
    except ImportError:
        return None

    ranges: list[tuple[int, int]] = []
    try:
        with h5py.File(path, "r") as handle:
            targets: list[str] = []

            def resolve(name: str) -> str | None:
                for candidate in (name, f"layers/{name}"):
                    if candidate in handle:
                        return candidate
                return None

            for name in layer_names:
                resolved = resolve(name)
                if resolved is None:
                    # A required layer is absent; the caller's assumptions do
                    # not hold for this file, so do not guess at a subset.
                    return None
                targets.append(resolved)

            # Variant-named layers differ between sources, so an absent one is
            # expected rather than a sign the layout is unrecognised.
            for name in optional_layer_names:
                resolved = resolve(name)
                if resolved is not None:
                    targets.append(resolved)

            for group_name in ("obs", "var"):
                group = handle.get(group_name)
                if group is None:
                    continue
                group.visit(
                    lambda sub, g=group, prefix=group_name: targets.append(
                        f"{prefix}/{sub}"
                    )
                    if isinstance(g.get(sub), h5py.Dataset)
                    else None
                )

            for target in targets:
                node = handle.get(target)
                if not isinstance(node, h5py.Dataset):
                    continue
                ranges.extend(_dataset_byte_ranges(node))
    except (OSError, RuntimeError, KeyError):
        return None

    if not ranges:
        return None
    return _merge_ranges(ranges)


def prewarm_h5ad(
    path: str | os.PathLike[str],
    *,
    layer_names: Sequence[str] = ("logFC",),
    optional_layer_names: Sequence[str] = (),
    label: str | None = None,
    mode: str | None = None,
    verbose: bool = True,
) -> PrewarmResult:
    """Warm only the layers a scan will actually read, plus obs/var.

    These files hold eleven same-shaped layers, so warming the whole file
    reads about eleven times the bytes the caller needs.  When the layout can
    be resolved this warms just the relevant spans; otherwise it falls back to
    :func:`prewarm_file`.

    Resolving the spans reads each layer's chunk index. On a cold disk that
    alone takes tens of seconds per layer for the largest sources, and every
    scoring worker opens every source at the start of every task. So the
    already-warmed checks -- this process first, then the cross-process lock
    record -- come before any index work, and only the one process that
    actually warms a file ever reads its index here.
    """
    path = Path(path)
    mode = (mode or prewarm_mode()).lower()
    display = label or path.name

    if mode == MODE_OFF:
        return PrewarmResult(path, mode, 0, 0, 0.0, "disabled")

    identity = _file_identity(path)
    if identity is None:
        return PrewarmResult(path, mode, 0, 0, 0.0, "unreadable")
    file_bytes = identity[1]

    with _warmed_lock:
        if identity in _warmed_paths:
            return PrewarmResult(path, mode, file_bytes, 0, 0.0, "already warmed")
        _warmed_paths.add(identity)

    block_size = _block_size()
    threads = _thread_count()

    def _run() -> PrewarmResult:
        started = time.monotonic()
        lock = _PrewarmLock(str(path), _lock_timeout_seconds())
        if not lock.acquire():
            return PrewarmResult(
                path,
                mode,
                file_bytes,
                0,
                time.monotonic() - started,
                "timed out waiting for another process to warm it",
            )
        fall_back = False
        try:
            already = lock.completed_bytes(identity)
            if already > 0:
                result = PrewarmResult(
                    path,
                    mode,
                    already,
                    already,
                    time.monotonic() - started,
                    "warmed by another process",
                )
                if verbose:
                    print(f"[prewarm] {display}: {result.describe()}", flush=True)
                return result

            ranges = _h5ad_prewarm_ranges(path, layer_names, optional_layer_names)
            if ranges is None:
                fall_back = True
            else:
                wanted = sum(length for _, length in ranges)
                budget = _budget_bytes()
                if wanted > budget:
                    # Keep whole ranges rather than truncating mid-layer, so
                    # what is warmed stays useful.
                    kept: list[tuple[int, int]] = []
                    running = 0
                    for offset, length in ranges:
                        if running + length > budget:
                            break
                        kept.append((offset, length))
                        running += length
                    ranges = kept or [(ranges[0][0], budget)]
                    wanted = sum(length for _, length in ranges)
                    if verbose:
                        print(
                            f"[prewarm] {display}: layers exceed the "
                            f"{budget / 1e9:.1f} GB cache budget; warming "
                            f"{wanted / 1e9:.1f} GB",
                            flush=True,
                        )
                try:
                    read_bytes = _parallel_read(path, ranges, block_size, threads)
                except OSError as exc:
                    with _warmed_lock:
                        _warmed_paths.discard(identity)
                    return PrewarmResult(
                        path, mode, wanted, 0, time.monotonic() - started, str(exc)
                    )
                lock.record(identity, read_bytes)
        finally:
            lock.release()

        if fall_back:
            # Released first: prewarm_file takes the same per-path lock.
            with _warmed_lock:
                _warmed_paths.discard(identity)
            return prewarm_file(path, label=label, mode=MODE_BLOCKING, verbose=verbose)

        result = PrewarmResult(
            path, mode, wanted, read_bytes, time.monotonic() - started
        )
        if verbose:
            print(
                f"[prewarm] {display}: {result.describe()} "
                f"of {file_bytes / 1e9:.1f} GB on disk",
                flush=True,
            )
        return result

    if mode == MODE_BACKGROUND:
        if verbose:
            print(f"[prewarm] {display}: warming layers in the background", flush=True)
        threading.Thread(
            target=_run, name=f"prewarm-{display}", daemon=True
        ).start()
        return PrewarmResult(path, mode, file_bytes, 0, 0.0, "started in background")

    return _run()


def prewarm_files(
    paths: Iterable[str | os.PathLike[str]],
    *,
    mode: str | None = None,
    verbose: bool = True,
) -> list[PrewarmResult]:
    """Warm several files in order, sharing one budget decision per file."""
    return [
        prewarm_file(path, mode=mode, verbose=verbose)
        for path in paths
    ]


def _self_test() -> None:
    import random

    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "sample.bin"
        payload = bytes(random.getrandbits(8) for _ in range(3 * 1024 * 1024))
        sample.write_bytes(payload)

        full = prewarm_file(sample, verbose=False)
        assert full.complete, full
        assert full.warmed_bytes == len(payload), full

        repeat = prewarm_file(sample, verbose=False)
        assert repeat.skipped_reason == "already warmed", repeat

        other = Path(tmp) / "capped.bin"
        other.write_bytes(payload)
        capped = prewarm_file(
            other, budget_bytes=_MIN_BUDGET_BYTES, verbose=False
        )
        # A 3 MB file is under any floor, so the cap must not truncate it.
        assert capped.complete, capped

        missing = prewarm_file(Path(tmp) / "absent.bin", verbose=False)
        assert missing.skipped_reason == "unreadable", missing

        disabled = prewarm_file(sample, mode=MODE_OFF, verbose=False)
        assert disabled.skipped_reason == "disabled", disabled

        background = Path(tmp) / "background.bin"
        background.write_bytes(payload)
        launched = prewarm_file(background, mode=MODE_BACKGROUND, verbose=False)
        assert launched.skipped_reason == "started in background", launched

        # A second process seeing a recorded warm must not re-read the file.
        shared = Path(tmp) / "shared.bin"
        shared.write_bytes(payload)
        identity = _file_identity(shared)
        assert identity is not None
        holder = _PrewarmLock(str(shared), 5.0)
        assert holder.acquire()
        try:
            assert holder.completed_bytes(identity) == 0
            holder.record(identity, len(payload))
            assert holder.completed_bytes(identity) == len(payload)
        finally:
            holder.release()
        reused = prewarm_file(shared, verbose=False)
        assert reused.skipped_reason == "warmed by another process", reused

        # A record from an earlier boot must be ignored: the page cache is gone.
        stale_boot = _PrewarmLock(str(shared), 5.0)
        assert stale_boot.acquire()
        try:
            os.lseek(stale_boot._fd, 0, os.SEEK_SET)
            os.ftruncate(stale_boot._fd, 0)
            os.write(stale_boot._fd, f"other-boot:{identity[1]}:{identity[2]}:9".encode())
            assert stale_boot.completed_bytes(identity) == 0
            stale_boot.record(identity, len(payload))
        finally:
            stale_boot.release()

        # A stale record from a different file revision must be ignored.
        stale = _PrewarmLock(str(shared), 5.0)
        assert stale.acquire()
        try:
            assert stale.completed_bytes((str(shared), 1, 1)) == 0
        finally:
            stale.release()

        # Losing the race past the timeout reports rather than reading cold.
        contended = Path(tmp) / "contended.bin"
        contended.write_bytes(payload)
        blocker = _PrewarmLock(str(contended), 5.0)
        assert blocker.acquire()
        try:
            os.environ["CPB_PAGE_CACHE_PREWARM_LOCK_TIMEOUT_S"] = "0"
            timed_out = prewarm_file(contended, verbose=False)
            assert timed_out.skipped_reason == (
                "timed out waiting for another process to warm it"
            ), timed_out
        finally:
            os.environ.pop("CPB_PAGE_CACHE_PREWARM_LOCK_TIMEOUT_S", None)
            blocker.release()

        _self_test_h5ad(Path(tmp))

    assert prewarm_mode() in _VALID_MODES
    print("page_cache self-tests passed")


def _self_test_h5ad(tmp: Path) -> None:
    try:
        import h5py
        import numpy as np
    except ImportError:
        print("page_cache: skipping H5AD self-test (h5py unavailable)")
        return

    path = tmp / "layers.h5ad"
    rows, cols = 600, 800
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as handle:
        layers = handle.create_group("layers")
        for name in ("logFC", "t", "P.Value"):
            layers.create_dataset(
                name,
                data=rng.normal(size=(rows, cols)),
                chunks=(100, 200),
                compression="gzip",
            )
        handle.create_group("obs").create_dataset(
            "pert_dose_uM", data=rng.normal(size=rows)
        )
        handle.create_group("var").create_dataset(
            "is_merged", data=np.ones(cols, dtype=bool)
        )

    file_bytes = path.stat().st_size
    ranges = _h5ad_prewarm_ranges(path, ("logFC",))
    assert ranges is not None, "expected resolvable layout"
    wanted = sum(length for _, length in ranges)
    # One of three same-shaped layers plus small metadata must be well under
    # the whole file; otherwise the scoping is not actually saving reads.
    assert wanted < file_bytes * 0.7, (wanted, file_bytes)

    warmed = prewarm_h5ad(path, layer_names=("logFC",), verbose=False)
    assert warmed.skipped_reason is None, warmed
    assert warmed.warmed_bytes == wanted, (warmed.warmed_bytes, wanted)

    # An absent layer means the caller's assumptions do not hold, so the
    # whole-file path must take over rather than warming a guessed subset.
    other = tmp / "layers_copy.h5ad"
    other.write_bytes(path.read_bytes())
    assert _h5ad_prewarm_ranges(other, ("missing_layer",)) is None
    fallback = prewarm_h5ad(other, layer_names=("missing_layer",), verbose=False)
    assert fallback.complete, fallback
    assert fallback.warmed_bytes == other.stat().st_size, fallback

    # An optional layer that exists is included; one that does not is skipped
    # without falling back to the whole file.
    with_optional = _h5ad_prewarm_ranges(path, ("logFC",), ("t", "absent"))
    assert with_optional is not None
    assert sum(n for _, n in with_optional) > wanted, (with_optional, wanted)

    merged = _merge_ranges([(0, 10), (8, 10), (100, 5)], gap_bytes=0)
    assert merged == [(0, 18), (100, 5)], merged
    pieces = _split_ranges([(0, 25)], 10)
    assert pieces == [(0, 10), (10, 10), (20, 5)], pieces


if __name__ == "__main__":
    _self_test()
