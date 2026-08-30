#!/usr/bin/env python
"""Add PI0.5-required quantiles to a local LeRobot dataset without network writes."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from lerobot.datasets import DEFAULT_QUANTILES, LeRobotDataset, write_stats


REQUIRED_QUANTILES = tuple(f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES)


def missing_quantiles(stats: dict, feature_names: list[str]) -> dict[str, list[str]]:
    return {
        feature: [key for key in REQUIRED_QUANTILES if key not in stats.get(feature, {})]
        for feature in feature_names
        if any(key not in stats.get(feature, {}) for key in REQUIRED_QUANTILES)
    }


def compute_exact_quantiles(dataset: LeRobotDataset, feature_names: list[str]) -> dict:
    stats = {feature: dict(feature_stats) for feature, feature_stats in dataset.meta.stats.items()}
    for feature in feature_names:
        if feature not in dataset.features:
            raise KeyError(f"Dataset does not contain required feature: {feature}")
        column = dataset.hf_dataset.data.column(feature).combine_chunks()
        if not hasattr(column, "values"):
            raise TypeError(f"Expected a list-like Arrow column for {feature}, got {column.type}")
        values = column.values.to_numpy(zero_copy_only=False).reshape(len(column), -1)
        print(f"Computing {feature} quantiles from Arrow column {column.type}, shape={values.shape}")
        for quantile in DEFAULT_QUANTILES:
            stats[feature][f"q{int(quantile * 100):02d}"] = np.quantile(values, quantile, axis=0)
    return stats


def write_stats_atomically(stats: dict, dataset_root: Path) -> None:
    stats_path = dataset_root / "meta" / "stats.json"
    backup_path = dataset_root / "meta" / "stats.pre_quantiles.json"
    if not backup_path.exists():
        shutil.copy2(stats_path, backup_path)

    with tempfile.TemporaryDirectory(dir=dataset_root) as temporary_directory:
        temporary_root = Path(temporary_directory)
        write_stats(stats, temporary_root)
        os.replace(temporary_root / "meta" / "stats.json", stats_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--feature", action="append", dest="features", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    missing = missing_quantiles(dataset.meta.stats, args.features)
    if args.check_only:
        if missing:
            raise SystemExit(f"Missing required quantiles: {missing}")
        print(f"Quantile contract satisfied for: {', '.join(args.features)}")
        return

    if not missing:
        print(f"Quantile contract already satisfied for: {', '.join(args.features)}")
        return

    updated_stats = compute_exact_quantiles(dataset, args.features)
    write_stats_atomically(updated_stats, args.root)
    remaining = missing_quantiles(updated_stats, args.features)
    if remaining:
        raise RuntimeError(f"Quantile augmentation incomplete: {remaining}")
    print(f"Added exact quantiles for: {', '.join(args.features)}")


if __name__ == "__main__":
    main()
