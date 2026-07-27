import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.notebook_cache import (
    CACHE_STATUS,
    FORCE_RECOMPUTE,
    IncrementalContextFrameStore,
    ProgressReporter,
    cached_frame,
    file_inventory_fingerprint,
    format_progress,
    frame_identity_fingerprint,
    resumable_context_frame,
    stable_json_fingerprint,
)


class MutableClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class NotebookCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        FORCE_RECOMPUTE.clear()
        CACHE_STATUS.clear()

    def tearDown(self) -> None:
        FORCE_RECOMPUTE.clear()
        CACHE_STATUS.clear()

    def test_stable_fingerprints_ignore_mapping_set_and_default_frame_order(self) -> None:
        left = {"datasets": {"tahoe", "sciplex"}, "settings": {"seed": 123, "cap": 512}}
        right = {"settings": {"cap": 512, "seed": 123}, "datasets": {"sciplex", "tahoe"}}
        self.assertEqual(
            stable_json_fingerprint(left),
            stable_json_fingerprint(right),
        )

        frame = pd.DataFrame(
            {
                "dataset_a": ["tahoe", "l1000_phase1"],
                "left_obs_id": ["002", "001"],
            }
        )
        reversed_frame = frame.iloc[::-1].reset_index(drop=True)
        self.assertEqual(
            frame_identity_fingerprint(frame, ["dataset_a", "left_obs_id"]),
            frame_identity_fingerprint(
                reversed_frame,
                ["dataset_a", "left_obs_id"],
            ),
        )
        self.assertNotEqual(
            frame_identity_fingerprint(
                frame,
                ["dataset_a", "left_obs_id"],
                order_sensitive=True,
            ),
            frame_identity_fingerprint(
                reversed_frame,
                ["dataset_a", "left_obs_id"],
                order_sensitive=True,
            ),
        )

    def test_file_inventory_and_cached_frame_invalidate_after_input_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("first")
            first_fingerprint = file_inventory_fingerprint(
                [source],
                root=root,
                payload={"datasets": ["tahoe"]},
                include_content_hash=True,
            )
            path = root / "result.tsv"
            calls = []

            def build() -> pd.DataFrame:
                calls.append(len(calls))
                return pd.DataFrame({"value": [len(calls)]})

            first = cached_frame(
                "inventory",
                path,
                build,
                fingerprint=first_fingerprint,
                verbose=False,
            )
            hit = cached_frame(
                "inventory",
                path,
                build,
                fingerprint=first_fingerprint,
                verbose=False,
            )
            self.assertEqual(calls, [0])
            self.assertEqual(first["value"].tolist(), hit["value"].tolist())
            self.assertEqual(CACHE_STATUS["inventory"], "reloaded")

            source.write_text("second")
            second_fingerprint = file_inventory_fingerprint(
                [source],
                root=root,
                payload={"datasets": ["tahoe"]},
                include_content_hash=True,
            )
            self.assertNotEqual(first_fingerprint, second_fingerprint)
            rebuilt = cached_frame(
                "inventory",
                path,
                build,
                fingerprint=second_fingerprint,
                verbose=False,
            )
            self.assertEqual(calls, [0, 1])
            self.assertEqual(rebuilt["value"].tolist(), [2])
            self.assertEqual(CACHE_STATUS["inventory"], "computed")

    def test_resumable_context_frame_hits_and_resumes_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard_root = Path(directory) / "shards"
            calls = []
            should_fail = {"b": True}
            observation_ids = {"a": "001", "b": "002", "c": "003"}

            def build(context: str) -> pd.DataFrame:
                calls.append(context)
                if context == "b" and should_fail["b"]:
                    raise RuntimeError("simulated interruption")
                return pd.DataFrame(
                    {
                        "left_obs_id": [observation_ids[context]],
                        "score": [ord(context)],
                    }
                )

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                resumable_context_frame(
                    "peer_contexts",
                    shard_root,
                    ["a", "b", "c"],
                    lambda context: {"context": context},
                    build,
                    fingerprint="peer-v1",
                    required_columns=["left_obs_id", "score"],
                    verbose=False,
                )
            self.assertEqual(calls, ["a", "b"])

            should_fail["b"] = False
            resumed = resumable_context_frame(
                "peer_contexts",
                shard_root,
                ["a", "b", "c"],
                lambda context: {"context": context},
                build,
                fingerprint="peer-v1",
                required_columns=["left_obs_id", "score"],
                verbose=False,
            )
            self.assertEqual(calls, ["a", "b", "b", "c"])
            self.assertEqual(resumed["left_obs_id"].tolist(), ["001", "002", "003"])
            self.assertEqual(CACHE_STATUS["peer_contexts"], "resumed")

            def explode(context: str) -> pd.DataFrame:
                raise AssertionError(f"cache miss for {context}")

            hit = resumable_context_frame(
                "peer_contexts",
                shard_root,
                ["c", "b", "a"],
                lambda context: {"context": context},
                explode,
                fingerprint="peer-v1",
                required_columns=["left_obs_id", "score"],
                verbose=False,
            )
            self.assertEqual(hit["left_obs_id"].tolist(), ["003", "002", "001"])
            self.assertEqual(CACHE_STATUS["peer_contexts"], "reloaded")

            invalidated_calls = []
            rebuilt = resumable_context_frame(
                "peer_contexts",
                shard_root,
                ["a", "b", "c"],
                lambda context: {"context": context},
                lambda context: (
                    invalidated_calls.append(context),
                    pd.DataFrame({"left_obs_id": [context], "score": [1]}),
                )[1],
                fingerprint="peer-v2",
                required_columns=["left_obs_id", "score"],
                verbose=False,
            )
            self.assertEqual(invalidated_calls, ["a", "b", "c"])
            self.assertEqual(rebuilt["left_obs_id"].tolist(), ["a", "b", "c"])
            self.assertEqual(CACHE_STATUS["peer_contexts"], "computed")

    def test_incremental_context_store_skips_saved_contexts_and_assembles_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keys = [("A", "B", "line1"), ("A", "C", "line2")]
            first = IncrementalContextFrameStore(
                "retrieval",
                root,
                keys,
                fingerprint="retrieval-v1",
            )
            self.assertFalse(first.is_complete(keys[0]))
            first.save(
                keys[0],
                pd.DataFrame({"query_obs_id": ["001"], "score": [0.5]}),
            )
            self.assertTrue(first.is_complete(keys[0]))
            with self.assertRaisesRegex(RuntimeError, "has not been checkpointed"):
                first.assemble()

            resumed = IncrementalContextFrameStore(
                "retrieval",
                root,
                keys,
                fingerprint="retrieval-v1",
            )
            self.assertTrue(resumed.is_complete(keys[0]))
            self.assertFalse(resumed.is_complete(keys[1]))
            resumed.save(
                keys[1],
                pd.DataFrame({"query_obs_id": ["002"], "score": [0.7]}),
            )
            assembled = resumed.assemble()
            self.assertEqual(assembled["query_obs_id"].tolist(), ["001", "002"])

            invalidated = IncrementalContextFrameStore(
                "retrieval",
                root,
                keys,
                fingerprint="retrieval-v2",
            )
            self.assertFalse(invalidated.is_complete(keys[0]))
            self.assertFalse(invalidated.is_complete(keys[1]))

    def test_progress_format_and_rate_limited_reporter(self) -> None:
        self.assertEqual(
            format_progress(
                label="peer_baselines",
                completed=25,
                total=100,
                elapsed_seconds=10,
                detail="Tahoe vs sci-Plex",
            ),
            (
                "[peer_baselines] 25/100 (25.0%) | elapsed 00:10 | "
                "2.50/s | ETA 00:30 | Tahoe vs sci-Plex"
            ),
        )

        clock = MutableClock()
        stream = io.StringIO()
        reporter = ProgressReporter(
            total=4,
            label="retrieval",
            every=10,
            min_interval_seconds=5,
            stream=stream,
            time_fn=clock,
        )
        clock.now = 1
        self.assertIsNone(reporter.update(detail="first"))
        clock.now = 6
        message = reporter.update(detail="second")
        self.assertIn("2/4 (50.0%)", message)
        self.assertIn("ETA 00:06", message)
        clock.now = 8
        reporter.update(completed=4, detail="done")
        output = stream.getvalue()
        self.assertEqual(output.count("\n"), 2)
        self.assertIn("[retrieval] 4/4 (100.0%)", output)


if __name__ == "__main__":
    unittest.main()
