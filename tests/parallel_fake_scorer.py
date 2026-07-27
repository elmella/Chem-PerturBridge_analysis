"""Spawn-importable test scorer for cross_source_parallel."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from scripts.cross_source_parallel import (
    TaskSpec,
    diagnostic_frame,
    read_task_input,
    report_task_progress,
)


def score_task(
    task: TaskSpec,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(config.get("fail_task_id", -1)) == task.task_id:
        raise RuntimeError(f"injected task failure {task.task_id}")
    if config.get("emit_progress"):
        report_task_progress(
            config,
            task,
            phase="fixture_work",
            detail="testing worker milestone",
        )
    frame = read_task_input(config, task)
    result = frame.copy()
    result["task_id"] = task.task_id
    result["doubled"] = pd.to_numeric(result["value"]) * 2
    if task.task_id in {
        int(value) for value in config.get("optional_metric_task_ids", [])
    }:
        result["optional_metric"] = result["doubled"] + 100
    if task.task_id in {
        int(value) for value in config.get(
            "incompatible_metric_task_ids",
            [],
        )
    }:
        result["different_optional_metric"] = result["doubled"] + 200
    return result, diagnostic_frame([])
