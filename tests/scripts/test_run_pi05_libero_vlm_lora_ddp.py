#!/usr/bin/env python

import os
import subprocess
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).parents[2] / "scripts" / "run_pi05_libero_vlm_lora_ddp.sh"


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


class Pi05LiberoVlmLoraLauncherTest(unittest.TestCase):
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

    def test_default_scope_targets_language_backbone_only(self) -> None:
        result = dry_run(8, "--lora-r", "32", "--lora-alpha", "64")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("distributed strategy: DDP", result.stdout)
        self.assertIn("--nproc-per-node=8", result.stdout)
        self.assertIn("--peft.method_type=LORA", result.stdout)
        self.assertIn("--peft.r=32", result.stdout)
        self.assertIn("--peft.lora_alpha=64", result.stdout)
        self.assertIn("LoRA scope: language", result.stdout)
        self.assertIn("Gemma-2B language backbone q/v projections (18 layers)", result.stdout)
        self.assertIn("Action Expert: frozen", result.stdout)
        self.assertIn("language_model", result.stdout)
        self.assertNotIn("gemma_expert", result.stdout)
        self.assertNotIn("vision_tower", result.stdout)
        self.assertIn("--policy.freeze_vision_encoder=true", result.stdout)
        self.assertIn("--policy.train_expert_only=false", result.stdout)
        self.assertIn(
            "/pi05_expert_ddp_b8_30k/checkpoints/030000/pretrained_model", result.stdout
        )
        self.assertNotIn("--parallelism.force_fsdp", result.stdout)
        self.assertNotIn("--parallelism.dp_shard", result.stdout)
        self.assertNotIn("--accelerator.activation_checkpointing.mode", result.stdout)
        self.assertIn("--checkpoint_format=safetensors", result.stdout)

    def test_language_vision_scope_adds_siglip_targets(self) -> None:
        result = dry_run(8, "--scope", "language-vision")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LoRA scope: language-vision", result.stdout)
        self.assertIn("SigLIP q/v (27 layers)", result.stdout)
        self.assertIn("language_model", result.stdout)
        self.assertIn("vision_tower", result.stdout)
        self.assertNotIn("gemma_expert", result.stdout)
        self.assertIn("--policy.freeze_vision_encoder=false", result.stdout)

    def test_invalid_scope_fails_before_launch(self) -> None:
        result = dry_run(8, "--scope", "expert")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--scope must be 'language' or 'language-vision'", result.stderr)

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
        result = dry_run(8, "--exp-name", "lora_resume", "--resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume=true", result.stdout)
        self.assertIn(
            "lora_resume/checkpoints/last/pretrained_model/train_config.json", result.stdout
        )
        self.assertNotIn("--policy.path=", result.stdout)
        self.assertNotIn("base policy:", result.stdout.split("command:", 1)[1])

    def test_resume_and_overwrite_are_mutually_exclusive(self) -> None:
        result = dry_run(8, "--resume", "--overwrite")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutually exclusive", result.stderr)


if __name__ == "__main__":
    unittest.main()
