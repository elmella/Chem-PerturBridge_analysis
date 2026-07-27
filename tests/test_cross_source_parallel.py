from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

from scripts.cross_source_parallel import (
    build_tasks,
    materialize_task_plan,
    output_directory_lock,
    run_checkpointed_tasks,
    task_paths,
    validate_checkpoint,
)


class CrossSourceParallelTests(unittest.TestCase):
    def test_output_directory_allows_only_one_coordinator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with output_directory_lock(root):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Another scoring coordinator",
                ):
                    with output_directory_lock(root):
                        pass

    def _plan(self, root: Path):
        frame = pd.DataFrame(
            {
                "context": ["b", "a", "b", "a"],
                "value": [3, 1, 4, 2],
            }
        )
        tasks, task_frames = build_tasks(
            frame,
            context_columns=["context"],
            rows_per_shard=1,
        )
        materialize_task_plan(
            tasks=tasks,
            task_frames=task_frames,
            output_dir=root,
        )
        config = {"output_dir": str(root)}
        return tasks, config

    def _run(
        self,
        root: Path,
        *,
        workers: int,
        force: bool = False,
        config_updates=None,
        progress_mode: str = "auto",
    ):
        tasks, config = self._plan(root)
        config.update(config_updates or {})
        outputs = run_checkpointed_tasks(
            analysis="fake",
            scorer_module="tests.parallel_fake_scorer",
            tasks=tasks,
            worker_config=config,
            output_dir=root,
            fingerprint="fake-v1",
            workers=workers,
            force=force,
            final_metrics_name="fake.tsv",
            progress_mode=progress_mode,
        )
        return tasks, outputs

    def test_serial_and_parallel_results_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            serial_root = root / "serial"
            parallel_root = root / "parallel"
            _, (serial_path, _) = self._run(serial_root, workers=1)
            _, (parallel_path, _) = self._run(parallel_root, workers=2)
            self.assertEqual(
                serial_path.read_bytes(),
                parallel_path.read_bytes(),
            )
            result = pd.read_csv(serial_path, sep="\t")
            self.assertEqual(result["context"].tolist(), ["b", "b", "a", "a"])
            self.assertEqual(result["doubled"].tolist(), [6, 8, 2, 4])

    def test_run_persists_incremental_progress_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._run(root, workers=1)
            progress_log = root / "progress.log"
            self.assertTrue(progress_log.is_file())
            contents = progress_log.read_text()
            self.assertIn("[fake] tasks=4; cached=0; pending=4; workers=1", contents)
            self.assertIn("[fake] completed 4/4 task=4 rows=1", contents)
            self.assertIn("[fake] merged 4 rows", contents)
            run_manifest = json.loads(
                (root / "run_manifest.json").read_text()
            )
            self.assertEqual(run_manifest["progress_log_file"], "progress.log")

    def test_always_progress_mode_renders_completed_tqdm_bar(self):
        with tempfile.TemporaryDirectory() as directory:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self._run(
                    Path(directory),
                    workers=1,
                    progress_mode="always",
                )
            rendered = stderr.getvalue()
            self.assertIn("fake", rendered)
            self.assertIn("4/4", rendered)
            self.assertIn("100%", rendered)

    def test_worker_milestones_are_saved_in_the_shared_progress_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self._run(
                    root,
                    workers=1,
                    progress_mode="off",
                    config_updates={"emit_progress": True},
                )
            self.assertEqual(stdout.getvalue(), "")
            contents = (root / "progress.log").read_text()
            self.assertIn(
                "[fake] task=1 phase=fixture_work testing worker milestone",
                contents,
            )

    def test_resume_skips_valid_shards_and_repairs_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, _ = self._run(root, workers=1)
            checkpoint = root / "checkpoints" / "fake-v1"
            marker_mtimes = {
                task.task_id: task_paths(checkpoint, task)[2].stat().st_mtime_ns
                for task in tasks
            }
            time.sleep(0.01)
            self._run(root, workers=2)
            self.assertEqual(
                marker_mtimes,
                {
                    task.task_id: task_paths(checkpoint, task)[2].stat().st_mtime_ns
                    for task in tasks
                },
            )

            corrupt_task = tasks[1]
            metrics_path, _, marker_path = task_paths(checkpoint, corrupt_task)
            metrics_path.write_text("corrupt\n")
            self.assertIsNone(
                validate_checkpoint(
                    checkpoint,
                    corrupt_task,
                    analysis="fake",
                    fingerprint="fake-v1",
                )
            )
            previous_marker = marker_path.stat().st_mtime_ns
            time.sleep(0.01)
            self._run(root, workers=2)
            self.assertGreater(marker_path.stat().st_mtime_ns, previous_marker)
            self.assertIsNotNone(
                validate_checkpoint(
                    checkpoint,
                    corrupt_task,
                    analysis="fake",
                    fingerprint="fake-v1",
                    load=True,
                )
            )

    def test_worker_failure_preserves_completed_checkpoints_and_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, (final_path, _) = self._run(root, workers=1)
            previous_final = final_path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "injected task failure"):
                self._run(
                    root,
                    workers=1,
                    force=True,
                    config_updates={"fail_task_id": tasks[1].task_id},
                )
            self.assertEqual(final_path.read_bytes(), previous_final)


if __name__ == "__main__":
    unittest.main()
