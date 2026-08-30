#!/usr/bin/env python

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch
from accelerate import Accelerator
from torch import nn

from lerobot.common.train_utils import (
    load_training_metadata,
    resume_after_prepare,
    resume_before_prepare,
    save_checkpoint,
    save_training_metadata,
    should_save_checkpoint,
)
from lerobot.configs.accelerator import AcceleratorConfig, GradientAccumulationConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.random_utils import save_rng_state
from tests.fixtures.dummy_checkpoint_policy import make_dummy_policy


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        return (self.weight * batch).square(), None


class CountingSGD(torch.optim.SGD):
    def __init__(self, params) -> None:
        super().__init__(params, lr=0.1)
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)

    def zero_grad(self, set_to_none=True):
        self.zero_grad_calls += 1
        return super().zero_grad(set_to_none=set_to_none)


class CountingScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


class Metrics:
    def update_metrics(self, output_dict) -> None:
        pass


def capture_gradient_accumulation_plugin(
    parallelism: ParallelismConfig, steps: int
):
    captured = {}

    class FakeAccelerator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with mock.patch("accelerate.Accelerator", FakeAccelerator):
        AcceleratorConfig(
            gradient_accumulation=GradientAccumulationConfig(steps=steps)
        ).build(parallelism, cpu=True)
    return captured["gradient_accumulation_plugin"]


class TestGradientSyncMode(unittest.TestCase):
    def test_fsdp_accum16_syncs_each_batch(self) -> None:
        parallelism = ParallelismConfig(dp_shard=-1, force_fsdp=True)
        parallelism.resolve(8)
        plugin = capture_gradient_accumulation_plugin(parallelism, steps=16)
        self.assertTrue(plugin.sync_each_batch)

    def test_fsdp_accum1_does_not_need_memory_safe_override(self) -> None:
        parallelism = ParallelismConfig(dp_shard=-1, force_fsdp=True)
        parallelism.resolve(8)
        plugin = capture_gradient_accumulation_plugin(parallelism, steps=1)
        self.assertFalse(plugin.sync_each_batch)

    def test_ddp_accum16_keeps_default_no_sync_behavior(self) -> None:
        parallelism = ParallelismConfig()
        parallelism.resolve(8)
        plugin = capture_gradient_accumulation_plugin(parallelism, steps=16)
        self.assertFalse(plugin.sync_each_batch)


class TestGradientAccumulation(unittest.TestCase):
    def test_32_micro_steps_produce_two_updates_with_single_loss_scaling(self) -> None:
        plugin = GradientAccumulationConfig(steps=16).build_plugin(sync_each_batch=True)
        accelerator = Accelerator(cpu=True, gradient_accumulation_plugin=plugin)
        policy = TinyPolicy()
        raw_optimizer = CountingSGD(policy.parameters())
        policy, optimizer = accelerator.prepare(policy, raw_optimizer)
        scheduler = CountingScheduler()
        metrics = Metrics()
        micro_step = 0
        optimizer_step = 0
        saved_at = []

        for _ in range(32):
            update_policy(
                metrics,
                policy,
                torch.tensor(1.0),
                optimizer,
                grad_clip_norm=0,
                accelerator=accelerator,
                lr_scheduler=scheduler,
            )
            micro_step += 1
            if accelerator.sync_gradients:
                optimizer_step += 1
                if should_save_checkpoint(optimizer_step, save_freq=2, total_steps=4):
                    saved_at.append(optimizer_step)

        self.assertEqual(micro_step, 32)
        self.assertEqual(optimizer_step, 2)
        self.assertEqual(raw_optimizer.step_calls, 2)
        self.assertEqual(raw_optimizer.zero_grad_calls, 2)
        self.assertEqual(scheduler.step_calls, 2)
        self.assertEqual(saved_at, [2])
        # Two SGD updates are exactly 1.0 -> 0.8 -> 0.64 when Accelerator.backward applies
        # one loss / 16. A second manual division would leave the weight near 1.0.
        self.assertAlmostEqual(policy.weight.item(), 0.64, places=5)

    def test_resume_restores_update_and_micro_step_at_boundary(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            state_dir = checkpoint_dir / "training_state"
            state_dir.mkdir()
            parallelism = SimpleNamespace(
                dp_world_size=8,
                dp_replicate=1,
                dp_shard=8,
                is_sharded=True,
                context_parallel=SimpleNamespace(ring_degree=1, ulysses_degree=1),
            )
            cfg = SimpleNamespace(
                checkpoint_path=checkpoint_dir,
                batch_size=1,
                parallelism=parallelism,
                accelerator=SimpleNamespace(
                    gradient_accumulation=SimpleNamespace(steps=16)
                ),
            )
            save_training_metadata(2, state_dir, cfg, micro_step=32)
            save_rng_state(state_dir)

            self.assertEqual(resume_before_prepare(cfg), 2)
            metadata = load_training_metadata(state_dir)
            self.assertEqual(metadata["optimizer_step"], 2)
            self.assertEqual(metadata["micro_step"], 32)

    def test_resumed_progress_continues_from_two_to_four_updates(self) -> None:
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=16)
        policy = TinyPolicy()
        raw_optimizer = CountingSGD(policy.parameters())
        policy, optimizer = accelerator.prepare(policy, raw_optimizer)
        scheduler = CountingScheduler()
        metrics = Metrics()
        optimizer_step = 2
        micro_step = 32

        while optimizer_step < 4:
            update_policy(
                metrics,
                policy,
                torch.tensor(1.0),
                optimizer,
                grad_clip_norm=0,
                accelerator=accelerator,
                lr_scheduler=scheduler,
            )
            micro_step += 1
            if accelerator.sync_gradients:
                optimizer_step += 1

        self.assertEqual(optimizer_step, 4)
        self.assertEqual(micro_step, 64)
        self.assertEqual(raw_optimizer.step_calls, 2)
        self.assertEqual(scheduler.step_calls, 2)

    def test_checkpoint_round_trip_restores_optimizer_and_scheduler_at_update_two(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)
            cfg = TrainPipelineConfig(
                dataset=DatasetConfig(repo_id="lerobot/dummy"), batch_size=1
            )
            cfg.parallelism.resolve(1)
            cfg.accelerator.gradient_accumulation.steps = 16
            policy = make_dummy_policy()
            optimizer = torch.optim.Adam(policy.parameters(), lr=0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
            for _ in range(2):
                optimizer.step()
                scheduler.step()
            passthrough = SimpleNamespace(
                unwrap_model=lambda model: model,
                wait_for_everyone=lambda: None,
            )
            save_checkpoint(
                checkpoint_dir,
                step=2,
                micro_step=32,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=passthrough,
            )
            cfg.checkpoint_path = checkpoint_dir

            fresh_policy = make_dummy_policy()
            fresh_optimizer = torch.optim.Adam(fresh_policy.parameters(), lr=0.9)
            fresh_scheduler = torch.optim.lr_scheduler.StepLR(
                fresh_optimizer, step_size=1, gamma=0.5
            )
            self.assertEqual(resume_before_prepare(cfg), 2)
            resume_after_prepare(
                cfg,
                passthrough,
                fresh_policy,
                fresh_optimizer,
                fresh_scheduler,
            )
            self.assertEqual(fresh_scheduler.last_epoch, 2)
            self.assertEqual(fresh_scheduler.get_last_lr(), scheduler.get_last_lr())


if __name__ == "__main__":
    unittest.main()
