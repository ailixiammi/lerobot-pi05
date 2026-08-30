#!/usr/bin/env python

import os
import subprocess
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).parents[2] / "scripts" / "run_pi05_libero_lora_ddp.sh"


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


class Pi05LiberoLoraLauncherTest(unittest.TestCase):
    def test_per_device_batch_math(self) -> None:
        cases = [
            (8, 8, 1, "global/per-device batch: 8/1"),
            (8, 16, 2, "global/per-device batch: 16/1"),
            (4, 16, 1, "global/per-device batch: 16/4"),
        ]
        for gpus, global_batch, accumulation, expected in cases:
            with self.subTest(gpus=gpus, global_batch=global_batch, accumulation=accumulation):
                result = dry_run(
                    gpus,
                    "--global-batch-size",
                    str(global_batch),
                    "--gradient-accumulation-steps",
                    str(accumulation),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)

    def test_invalid_global_batch_fails_with_nearby_values(self) -> None:
        result = dry_run(8, "--global-batch-size", "10")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nearby legal values: 8 or 16", result.stderr)

    def test_launcher_uses_ddp_and_pi05_default_lora_targets(self) -> None:
        result = dry_run(8, "--lora-r", "32", "--lora-alpha", "64")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("distributed strategy: DDP", result.stdout)
        self.assertIn("--nproc-per-node=8", result.stdout)
        self.assertIn("--peft.method_type=LORA", result.stdout)
        self.assertIn("--peft.r=32", result.stdout)
        self.assertIn("--peft.lora_alpha=64", result.stdout)
        self.assertIn("Action Expert q/v + action input/output projections", result.stdout)
        self.assertNotIn("--parallelism.force_fsdp", result.stdout)
        self.assertNotIn("--parallelism.dp_shard", result.stdout)
        self.assertNotIn("--accelerator.activation_checkpointing.mode", result.stdout)
        self.assertIn("--checkpoint_format=safetensors", result.stdout)

    def test_optimizer_and_micro_step_calculation(self) -> None:
        result = dry_run(
            8,
            "--global-batch-size",
            "16",
            "--gradient-accumulation-steps",
            "2",
            "--steps",
            "10000",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("effective_global_batch: 16", result.stdout)
        self.assertIn("target_optimizer_steps: 10000", result.stdout)
        self.assertIn("expected_micro_steps: 20000", result.stdout)

    def test_resume_uses_adapter_checkpoint_and_excludes_base_path(self) -> None:
        result = dry_run(8, "--exp-name", "lora_resume", "--resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume=true", result.stdout)
        self.assertIn(
            "lora_resume/checkpoints/last/pretrained_model/train_config.json", result.stdout
        )
        self.assertNotIn("--policy.path=", result.stdout)

    def test_resume_and_overwrite_are_mutually_exclusive(self) -> None:
        result = dry_run(8, "--resume", "--overwrite")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutually exclusive", result.stderr)


if __name__ == "__main__":
    unittest.main()
