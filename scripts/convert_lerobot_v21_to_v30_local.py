#!/usr/bin/env python3
"""Convert a local LeRobot Dataset v2.1 staging tree to an independent v3 tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.scripts.convert_dataset_v21_to_v30 import (
    convert_data,
    convert_episodes_metadata,
    convert_info,
    convert_tasks,
    convert_videos,
    validate_local_dataset_version,
)


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def validate_paths(source: Path, output: Path) -> tuple[Path, Path]:
    source = resolved(source)
    output = resolved(output)
    if source == output:
        raise ValueError("source and output paths must differ")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"missing source metadata: {source / 'meta' / 'info.json'}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    return source, output


def convert(
    repo_id: str,
    source: Path,
    output: Path,
    data_file_size_in_mb: int,
    video_file_size_in_mb: int,
) -> None:
    source, output = validate_paths(source, output)
    validate_local_dataset_version(source)
    source_info = json.loads((source / "meta" / "info.json").read_text())

    output.mkdir(parents=True)
    try:
        convert_info(source, output, data_file_size_in_mb, video_file_size_in_mb)
        convert_tasks(source, output)
        episodes_metadata = convert_data(source, output, data_file_size_in_mb)
        videos_metadata = convert_videos(source, output, video_file_size_in_mb)
        convert_episodes_metadata(source, output, episodes_metadata, videos_metadata)

        output_info = json.loads((output / "meta" / "info.json").read_text())
        for field in ("total_episodes", "total_frames", "total_tasks"):
            if output_info[field] != source_info[field]:
                raise RuntimeError(
                    f"conversion changed {field}: {source_info[field]} -> {output_info[field]}"
                )
        if output_info["codebase_version"] != "v3.0":
            raise RuntimeError(f"expected Dataset v3.0, found {output_info['codebase_version']!r}")

        dataset = LeRobotDataset(repo_id, root=output, episodes=[0], download_videos=False)
        sample = dataset[0]
        print(f"Validated first sample with keys: {sorted(sample)}")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise

    print(f"Converted Dataset v3 tree: {output}")
    print(f"Dataset v2.1 staging tree remained unchanged: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-file-size-in-mb", type=int, default=100)
    parser.add_argument("--video-file-size-in-mb", type=int, default=500)
    args = parser.parse_args()
    if args.data_file_size_in_mb < 1 or args.video_file_size_in_mb < 1:
        parser.error("file size limits must be positive")
    convert(
        args.repo_id,
        args.source,
        args.output,
        args.data_file_size_in_mb,
        args.video_file_size_in_mb,
    )


if __name__ == "__main__":
    main()
