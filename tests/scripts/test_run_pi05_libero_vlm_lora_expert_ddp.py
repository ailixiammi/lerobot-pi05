#!/usr/bin/env python

import os
import subprocess
import unittest
from pathlib import Path

LAUNCHER = Path(__file__).parents[2] / "scripts" / "run_pi05_libero_vlm_lora_expert_ddp.sh"


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


class Pi05LiberoVlmLoraExpertLauncherTest(unittest.TestCase):
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
        self.assertIn("distributed strategy: DDP (VLM LoRA + fully trainable Action Expert)", result.stdout)
        self.assertIn("--nproc-per-node=8", result.stdout)
        self.assertIn("--peft.method_type=LORA", result.stdout)
        self.assertIn("--peft.r=32", result.stdout)
        self.assertIn("--peft.lora_alpha=64", result.stdout)
        self.assertIn("LoRA scope: language", result.stdout)
        self.assertIn("Gemma-2B language backbone q/v projections (18 layers)", result.stdout)
        self.assertIn("Action Expert: full training", result.stdout)
        target_modules = target_modules_argument(result.stdout)
        self.assertIn("language_model", target_modules)
        self.assertNotIn("gemma_expert", target_modules)
        self.assertNotIn("vision_tower", target_modules)
        self.assertIn("--policy.freeze_vision_encoder=true", result.stdout)
        self.assertIn("--policy.train_expert_only=false", result.stdout)
        self.assertIn("/lerobot-hf-assets/pi05_base", result.stdout)
        self.assertIn("--peft.full_training_modules=", result.stdout)
        self.assertIn("gemma_expert", result.stdout)
        self.assertIn("action_in_proj", result.stdout)
        self.assertIn("action_out_proj", result.stdout)
        self.assertIn("time_mlp_in", result.stdout)
        self.assertIn("time_mlp_out", result.stdout)
        self.assertNotIn("state_proj", result.stdout)
        self.assertNotIn("action_time_mlp", result.stdout)
        self.assertNotIn("--parallelism.force_fsdp", result.stdout)
        self.assertIn("--parallelism.dp_replicate=8", result.stdout)
        self.assertIn("--parallelism.dp_shard=1", result.stdout)
        self.assertNotIn("--accelerator.activation_checkpointing.mode", result.stdout)
        self.assertIn("--checkpoint_format=safetensors", result.stdout)

    def test_language_vision_scope_adds_siglip_targets(self) -> None:
        result = dry_run(8, "--scope", "language-vision")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LoRA scope: language-vision", result.stdout)
        self.assertIn("SigLIP q/v (27 layers)", result.stdout)
        target_modules = target_modules_argument(result.stdout)
        self.assertIn("language_model", target_modules)
        self.assertIn("vision_tower", target_modules)
        self.assertNotIn("gemma_expert", target_modules)
        self.assertIn("--policy.freeze_vision_encoder=false", result.stdout)
        self.assertIn("Action Expert: full training", result.stdout)

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
        result = dry_run(8, "--exp-name", "hybrid_resume", "--resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume=true", result.stdout)
        self.assertIn(
            "hybrid_resume/checkpoints/last/pretrained_model/train_config.json", result.stdout
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
