#!/usr/bin/env python3
"""Stage a tarball-only dataset into the layout the scorers expect.

CIGS, GDPx2, and DILImap are not redistributable, so they ship as tarballs in
``data/google-drive/`` rather than in the HuggingFace mirror under
``data/theislab_temp/``.  This command unpacks one into place, then checks that
the resolvers used by the population and replicate scorers can actually find
the result — layouts differ between archives, so discovery beats assuming.

Two reasons this is a script and not a ``tar`` one-liner:

* **Pre-warming.** ``tar`` reads its input as one buffered stream, which this
  volume serves at ~4 MB/s.  Warming the archive first with concurrent reads
  (~25 MB/s) leaves decompression CPU-bound instead of latency-bound. A
  measured ``tar -tzvf`` of a 1 GB archive took 4m15s cold.
* **Not deleting the irreplaceable copies.** GDPx2 and DILImap archives are
  mirrored on the ``gdrive-data`` remote, so they can be removed after
  unpacking.  The CIGS archives exist nowhere else and are refused.

Examples::

    uv run python scripts/stage_archived_dataset.py --dataset gdpx2 --delete-archive-after
    uv run python scripts/stage_archived_dataset.py --dataset cigs_mce --mode sep_rep
    uv run python scripts/stage_archived_dataset.py --list
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from page_cache import prewarm_file  # noqa: E402

ARCHIVE_ROOT = REPO_ROOT / "data" / "google-drive"
STAGE_ROOT = REPO_ROOT / "data" / "theislab_temp"

MODES = ("group_rep", "sep_rep")

# Archives with no second copy anywhere. CIGS is licence-restricted, so it is
# absent from both the HuggingFace mirror and the gdrive-data remote.
IRREPLACEABLE_PREFIXES = ("cigs_mce", "cigs_tcm")


@dataclass(frozen=True)
class ArchiveSpec:
    dataset: str
    mode: str
    archive: Path
    processed: Path | None

    @property
    def stage_dir(self) -> Path:
        return STAGE_ROOT / self.dataset / f"{self.mode}_extracted"

    @property
    def replaceable(self) -> bool:
        return not self.archive.name.startswith(IRREPLACEABLE_PREFIXES)


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def known_archives() -> list[ArchiveSpec]:
    """Every archive present on disk, with its processed-inventory partner."""
    datasets = {
        "cigs_mce": (
            "cigs_mce_deg_data_{mode}.tar.gz",
            "cigs_mce_processed.h5ad",
        ),
        "cigs_tcm": (
            "cigs_tcm_deg_data_{mode}.tar.gz",
            "cigs_tcm_processed.h5ad",
        ),
        "gdpx2": ("gdpx2_{mode}.tar.gz", "gdpx2_standardized_processed.h5ad"),
        "dilimap_train_val": (
            "dilimap_train_val_{mode}.tar.gz",
            "dilimap_train_val_assembled_processed.h5ad",
        ),
    }
    specs: list[ArchiveSpec] = []
    for dataset, (archive_template, processed_name) in datasets.items():
        for mode in MODES:
            archive = ARCHIVE_ROOT / archive_template.format(mode=mode)
            if not archive.is_file():
                continue
            specs.append(
                ArchiveSpec(
                    dataset=dataset,
                    mode=mode,
                    archive=archive,
                    processed=_first_existing(ARCHIVE_ROOT / processed_name),
                )
            )
    return specs


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def _de_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*_de.h5ad"))


def _resolves_for_scorers(spec: ArchiveSpec) -> tuple[bool, str]:
    """Check the real resolvers, not a reimplementation of them."""
    if spec.mode == "sep_rep":
        from precompute_replicate_signature_similarity import sep_rep_dataset_dir

        resolved = sep_rep_dataset_dir(spec.dataset, 0)
        ok = resolved.is_dir() and bool(list(resolved.glob("*_de.h5ad")))
        return ok, f"replicate scorer resolves {spec.dataset} -> {resolved}"

    from precompute_population_zscore import configured_dataset_dirs

    configured = dict(configured_dataset_dirs(REPO_ROOT))
    resolved = configured.get(spec.dataset)
    if resolved is None:
        return False, (
            f"population precompute does not list {spec.dataset}; expected a "
            f"directory it recognises to exist"
        )
    ok = resolved.is_dir() and bool(list(resolved.glob("*.h5ad")))
    return ok, f"population precompute resolves {spec.dataset} -> {resolved}"


def link_processed_inventory(spec: ArchiveSpec, *, dry_run: bool) -> Path | None:
    """Symlink the processed candidate-inventory H5AD next to the dataset.

    The replicate scorer scans this to decide which lines and compounds are
    candidates, and looks for it directly under ``<root>/<dataset>/``. A
    symlink keeps the single copy on disk.
    """
    if spec.processed is None:
        return None
    destination = STAGE_ROOT / spec.dataset / spec.processed.name
    if destination.exists() or destination.is_symlink():
        return destination
    print(f"[stage] linking inventory {destination.name} -> {spec.processed}")
    if dry_run:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(spec.processed.resolve())
    return destination


def extract(spec: ArchiveSpec, *, dry_run: bool) -> list[Path]:
    existing = _de_files(spec.stage_dir) if spec.stage_dir.is_dir() else []
    if existing:
        print(
            f"[stage] {spec.dataset}/{spec.mode}: already unpacked "
            f"({len(existing)} *_de.h5ad), skipping extraction"
        )
        return existing

    archive_bytes = spec.archive.stat().st_size
    free = _free_bytes(STAGE_ROOT)
    # These archives hold h5ads whose layers are already gzip-compressed, so
    # the ratio is ~1:1 and the archive size is a good size estimate.
    print(
        f"[stage] {spec.dataset}/{spec.mode}: {archive_bytes / 1e9:.1f} GB archive, "
        f"{free / 1e9:.1f} GB free"
    )
    if free < archive_bytes * 1.15:
        raise RuntimeError(
            f"Not enough free space to unpack {spec.archive.name}: needs about "
            f"{archive_bytes / 1e9:.1f} GB, {free / 1e9:.1f} GB free. Free space "
            f"or delete an already-scored dataset under {STAGE_ROOT}."
        )

    prewarm_file(spec.archive, label=f"{spec.dataset}/{spec.mode} archive")

    if dry_run:
        print(f"[stage] dry run: would extract into {spec.stage_dir}")
        return []

    # Unpack beside the target and move into place, so an interrupted run
    # cannot leave a half-populated directory that looks already-unpacked.
    staging = spec.stage_dir.with_name(spec.stage_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    started = time.monotonic()
    try:
        with tarfile.open(spec.archive, "r:gz") as handle:
            _safe_extract(handle, staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.replace(spec.stage_dir)
    elapsed = time.monotonic() - started
    found = _de_files(spec.stage_dir)
    print(
        f"[stage] {spec.dataset}/{spec.mode}: extracted {len(found)} *_de.h5ad "
        f"in {elapsed / 60:.1f} min ({archive_bytes / 1e6 / max(elapsed, 1e-9):.0f} MB/s)"
    )
    if not found:
        raise RuntimeError(
            f"{spec.archive.name} unpacked but contains no *_de.h5ad under "
            f"{spec.stage_dir}"
        )
    return found


def _safe_extract(handle: tarfile.TarFile, destination: Path) -> None:
    """Extract, refusing members that would escape ``destination``."""
    root = destination.resolve()
    for member in handle.getmembers():
        target = (root / member.name).resolve()
        if not str(target).startswith(str(root)):
            raise RuntimeError(f"Refusing unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing link member in archive: {member.name}")
    handle.extractall(root)


def delete_archive(spec: ArchiveSpec, *, dry_run: bool) -> bool:
    if not spec.replaceable:
        print(
            f"[stage] refusing to delete {spec.archive.name}: this is the only "
            "copy on this machine (CIGS is not on HuggingFace or the gdrive "
            "remote)"
        )
        return False
    freed = spec.archive.stat().st_size
    print(f"[stage] deleting {spec.archive.name}, freeing {freed / 1e9:.1f} GB")
    if not dry_run:
        spec.archive.unlink()
    return True


def run(args: argparse.Namespace) -> int:
    specs = known_archives()
    if args.list:
        print(f"{'dataset':<20} {'mode':<10} {'GB':>6}  refetchable  archive")
        for spec in specs:
            size = spec.archive.stat().st_size / 1e9
            print(
                f"{spec.dataset:<20} {spec.mode:<10} {size:>6.1f}  "
                f"{'yes' if spec.replaceable else 'NO   ':<11}  {spec.archive.name}"
            )
        return 0

    selected = [
        spec
        for spec in specs
        if (not args.dataset or spec.dataset in args.dataset)
        and (not args.mode or spec.mode in args.mode)
    ]
    if not selected:
        print("No matching archives found. Use --list to see what is available.")
        return 1

    failures: list[str] = []
    for spec in selected:
        print(f"\n=== {spec.dataset} / {spec.mode}")
        try:
            extract(spec, dry_run=args.dry_run)
            link_processed_inventory(spec, dry_run=args.dry_run)
            if not args.dry_run:
                ok, detail = _resolves_for_scorers(spec)
                print(f"[stage] {'OK  ' if ok else 'FAIL'} {detail}")
                if not ok:
                    failures.append(f"{spec.dataset}/{spec.mode}: {detail}")
                elif args.delete_archive_after:
                    delete_archive(spec, dry_run=args.dry_run)
        except Exception as exc:  # surface and continue to the next archive
            print(f"[stage] ERROR {spec.dataset}/{spec.mode}: {exc}")
            failures.append(f"{spec.dataset}/{spec.mode}: {exc}")

    print(f"\n[stage] free space now {_free_bytes(STAGE_ROOT) / 1e9:.1f} GB")
    if failures:
        print("[stage] problems:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset to stage (repeatable). Default: all available.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        choices=MODES,
        help="Replicate mode to stage (repeatable). Default: both.",
    )
    parser.add_argument(
        "--delete-archive-after",
        action="store_true",
        help=(
            "Delete the archive once it is unpacked and the scorers resolve it. "
            "Refused for the CIGS archives, which have no other copy."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without extracting or deleting.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the archives present and whether each can be refetched.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
