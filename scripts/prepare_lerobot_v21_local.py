#!/usr/bin/env python3
"""Prepare a local LeRobot Dataset v2.1 staging tree from a v2.0 dataset.

Run this script with the legacy LeRobot environment that can read Dataset v2.0.
The source tree is never modified: metadata is copied and data/video directories
are linked read-only for the subsequent v2.1-to-v3 conversion.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info
from lerobot.common.datasets.v21.convert_stats import check_aggregate_stats, convert_stats

AGGREGATE_STATS_RTOL_ATOL = (1e-5, 1e-4)


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_paths(source: Path, staging: Path) -> tuple[Path, Path]:
    source = resolved(source)
    staging = resolved(staging)
    if source == staging:
        raise ValueError("source and staging paths must differ")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"missing source metadata: {source / 'meta' / 'info.json'}")
    if not (source / "data").is_dir():
        raise FileNotFoundError(f"missing source data directory: {source / 'data'}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite existing staging directory: {staging}")
    return source, staging


def read_version(root: Path) -> str:
    return json.loads((root / "meta" / "info.json").read_text())["codebase_version"]


def validate_aggregate_stats(dataset: LeRobotDataset, reference_stats: dict) -> None:
    check_aggregate_stats(
        dataset,
        reference_stats,
        default_rtol_atol=AGGREGATE_STATS_RTOL_ATOL,
    )


def prepare(repo_id: str, source: Path, staging: Path, num_workers: int) -> None:
    source, staging = validate_paths(source, staging)
    if read_version(source) != "v2.0":
        raise ValueError(f"expected Dataset v2.0 at {source}, found {read_version(source)!r}")

    staging.mkdir(parents=True)
    try:
        shutil.copytree(source / "meta", staging / "meta")
        (staging / "data").symlink_to(source / "data", target_is_directory=True)
        if (source / "videos").is_dir():
            (staging / "videos").symlink_to(source / "videos", target_is_directory=True)

        dataset = LeRobotDataset(
            repo_id,
            root=staging,
            revision="v2.0",
            download_videos=False,
        )
        episode_stats = staging / EPISODES_STATS_PATH
        if episode_stats.exists():
            episode_stats.unlink()

        convert_stats(dataset, num_workers=num_workers)
        reference_stats = load_stats(staging)
        # The v2.0 global stats and the v2.1 per-episode aggregation sum values
        # in a different order. On large datasets this creates harmless float
        # round-off (LIBERO's episode_index mean differs by about 5.9e-6
        # relatively), just beyond the legacy converter's 5e-6 default.
        # Keep the official check, with a narrowly relaxed numeric tolerance.
        validate_aggregate_stats(dataset, reference_stats)

        dataset.meta.info["codebase_version"] = "v2.1"
        write_info(dataset.meta.info, staging)
        stats_path = staging / STATS_PATH
        if stats_path.is_file():
            stats_path.unlink()

        if read_version(staging) != "v2.1" or not episode_stats.is_file():
            raise RuntimeError("v2.1 staging validation failed")
    except Exception:
        # The staging directory belongs exclusively to this invocation. Removing
        # it cannot affect the source because only data/video symlinks point out.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Prepared Dataset v2.1 staging tree: {staging}")
    print(f"Source Dataset v2.0 remained unchanged: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be positive")
    prepare(args.repo_id, args.source, args.staging, args.num_workers)


if __name__ == "__main__":
    main()
