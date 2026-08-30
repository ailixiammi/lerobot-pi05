#!/usr/bin/env python

import os
import subprocess
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).parents[2] / "scripts" / "run_pi05_libero_openpi_lora_ddp.sh"


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


def target_modules_argument(output: str) -> str:
    return output.split("--peft.target_modules=", 1)[1].split(
        " --peft.full_training_modules=", 1
    )[0]


class Pi05LiberoOpenPiLoraLauncherTest(unittest.TestCase):
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
        self.assertIn("nearby legal values: 16", result.stderr)

    def test_default_run_matches_openpi_lora_topology(self) -> None:
        result = dry_run(8)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("distributed strategy: DDP (OpenPI-compatible LoRA)", result.stdout)
        self.assertIn("--nproc-per-node=8", result.stdout)
        self.assertIn("--peft.method_type=LORA", result.stdout)
        self.assertIn("--peft.r=16", result.stdout)
        self.assertIn("--peft.lora_alpha=16", result.stdout)
        self.assertIn("--peft.rank_pattern=", result.stdout)
        self.assertIn("--peft.alpha_pattern=", result.stdout)
        self.assertIn("PaliGemma LoRA: attention q/k/v/o + FFN gate/up/down", result.stdout)
        self.assertIn("Action Expert LoRA: attention q/k/v/o + FFN gate/up/down", result.stdout)
        target_modules = target_modules_argument(result.stdout)
        self.assertIn("language_model", target_modules)
        self.assertIn("gemma_expert", target_modules)
        self.assertIn(r"q\|k\|v\|o", target_modules)
        self.assertIn(r"gate\|up\|down", target_modules)
        self.assertNotIn("vision_tower", target_modules)
        self.assertIn("--policy.freeze_vision_encoder=true", result.stdout)
        self.assertIn("--policy.train_expert_only=false", result.stdout)
        self.assertIn("/lerobot-hf-assets/pi05_base", result.stdout)
        self.assertIn("--peft.full_training_modules=", result.stdout)
        self.assertIn("action_in_proj", result.stdout)
        self.assertIn("action_out_proj", result.stdout)
        self.assertIn("time_mlp_in", result.stdout)
        self.assertIn("time_mlp_out", result.stdout)
        self.assertNotIn("--parallelism.force_fsdp", result.stdout)
        self.assertIn("--parallelism.dp_replicate=8", result.stdout)
        self.assertIn("--parallelism.dp_shard=1", result.stdout)
        self.assertNotIn("--accelerator.activation_checkpointing.mode", result.stdout)
        self.assertIn("--checkpoint_format=safetensors", result.stdout)

    def test_default_full_run_saves_every_5000_steps(self) -> None:
        result = dry_run(8)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("global/per-device batch: 128/8 (gradient accumulation: 2)", result.stdout)
        self.assertIn("target_optimizer_steps: 30000", result.stdout)
        self.assertIn("--save_freq=5000", result.stdout)

    def test_custom_base_policy_is_forwarded(self) -> None:
        result = dry_run(8, "--base-policy", "/tmp/expert-checkpoint")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("base policy: /tmp/expert-checkpoint", result.stdout)
        self.assertIn("--policy.path=/tmp/expert-checkpoint", result.stdout)

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
        result = dry_run(8, "--exp-name", "openpi_lora_resume", "--resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume=true", result.stdout)
        self.assertIn(
            "openpi_lora_resume/checkpoints/last/pretrained_model/train_config.json", result.stdout
        )
        self.assertNotIn("--policy.path=", result.stdout)
        self.assertNotIn("base policy:", result.stdout.split("command:", 1)[1])
        self.assertIn("DDP topology: dp_replicate=8, dp_shard=1", result.stdout)

    def test_resume_overrides_saved_topology_with_current_gpu_count(self) -> None:
        result = dry_run(
            6,
            "--global-batch-size",
            "12",
            "--resume",
            "/tmp/four_gpu_checkpoint/train_config.json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--parallelism.dp_replicate=6", result.stdout)
        self.assertIn("--parallelism.dp_shard=1", result.stdout)

    def test_resume_and_overwrite_are_mutually_exclusive(self) -> None:
        result = dry_run(8, "--resume", "--overwrite")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutually exclusive", result.stderr)


if __name__ == "__main__":
    unittest.main()
