#!/usr/bin/env python

import os
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).parents[2] / "scripts" / "run_pi05_libero_fsdp.sh"


def dry_run(gpus: int, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(gpus))
    return subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


@pytest.mark.parametrize(
    ("gpus", "global_batch", "expected"),
    [
        (8, 8, "global/per-device batch: 8/1"),
        (4, 32, "global/per-device batch: 32/8"),
        (1, 8, "global/per-device batch: 8/8"),
    ],
)
def test_per_device_batch_math(gpus: int, global_batch: int, expected: str) -> None:
    result = dry_run(gpus, "--global-batch-size", str(global_batch))
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


def test_invalid_global_batch_fails_with_nearby_values() -> None:
    result = dry_run(8, "--global-batch-size", "10")
    assert result.returncode != 0
    assert "nearby legal values: 8 or 16" in result.stderr

    accumulated = dry_run(
        8,
        "--global-batch-size",
        "130",
        "--gradient-accumulation-steps",
        "16",
    )
    assert accumulated.returncode != 0
    assert "nearby legal values: 128 or 256" in accumulated.stderr


def test_optimizer_and_micro_step_calculation() -> None:
    result = dry_run(
        8,
        "--global-batch-size",
        "128",
        "--gradient-accumulation-steps",
        "16",
        "--steps",
        "30000",
    )
    assert result.returncode == 0, result.stderr
    assert "per_device_micro_batch: 1" in result.stdout
    assert "gradient_accumulation_steps: 16" in result.stdout
    assert "gradient_sync_mode=sync_each_batch" in result.stdout
    assert "FSDP memory-safe accumulation" in result.stdout
    assert "effective_global_batch: 128" in result.stdout
    assert "target_optimizer_steps: 30000" in result.stdout
    assert "expected_micro_steps: 480000" in result.stdout


def test_dry_run_is_full_parameter_fsdp_not_ddp() -> None:
    result = dry_run(8)
    assert result.returncode == 0, result.stderr
    assert "FSDP sharding strategy: FULL_SHARD" in result.stdout
    assert "--parallelism.dp_shard=-1" in result.stdout
    assert "--parallelism.force_fsdp=true" in result.stdout
    assert "--accelerator.fsdp.reshard_after_forward=true" in result.stdout
    assert "--accelerator.activation_checkpointing.mode=full" in result.stdout
    assert "Activation checkpointing mode: native_fsdp_full" in result.stdout
    assert "--policy.train_expert_only=false" in result.stdout
    assert "--policy.gradient_checkpointing=true" in result.stdout
    assert "DistributedDataParallel" not in result.stdout
    assert "expandable_segments" not in LAUNCHER.read_text()


def test_checkpoint_and_default_resume_configuration() -> None:
    result = dry_run(8, "--exp-name", "resume_test", "--save-freq", "5", "--resume")
    assert result.returncode == 0, result.stderr
    assert "--save_checkpoint=true" in result.stdout
    assert "--save_freq=5" in result.stdout
    assert "--checkpoint_format=dcp" in result.stdout
    assert "--resume=true" in result.stdout
    assert "resume_test/checkpoints/last/pretrained_model/train_config.json" in result.stdout
    assert "--policy.path=" not in result.stdout
