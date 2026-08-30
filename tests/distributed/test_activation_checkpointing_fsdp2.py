#!/usr/bin/env python

import os
import socket
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn

from lerobot.configs.accelerator import AcceleratorConfig
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.distributed import apply_fsdp_activation_checkpointing, set_fsdp_wrap_modules


class DtypeRecordingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.forward_dtypes: list[torch.dtype] = []

    def forward(self, value):
        self.forward_dtypes.append(value.dtype)
        return torch.nn.functional.gelu(self.proj(value))


class TinyCheckpointPolicy(nn.Module):
    _fsdp_wrap_modules = ["DtypeRecordingBlock"]

    def __init__(self) -> None:
        super().__init__()
        self.block = DtypeRecordingBlock()
        self.out = nn.Linear(8, 1)

    def forward(self, value):
        return self.out(self.block(value))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FSDP2")
class TestFSDP2BF16ActivationCheckpointing(unittest.TestCase):
    def test_forward_and_recompute_both_use_bf16(self) -> None:
        distributed_env = {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_free_port()),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }
        with mock.patch.dict(os.environ, distributed_env):
            torch.cuda.set_device(0)
            dist.init_process_group("nccl")
            try:
                parallelism = ParallelismConfig(dp_shard=-1, force_fsdp=True)
                parallelism.resolve(1)
                accelerator = AcceleratorConfig(mixed_precision="bf16").build(parallelism)
                policy = TinyCheckpointPolicy().cuda()
                optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

                set_fsdp_wrap_modules(accelerator, policy)
                wrapped_units = apply_fsdp_activation_checkpointing(accelerator, policy)
                self.assertEqual(wrapped_units, 1)
                policy, optimizer = accelerator.prepare(policy, optimizer)

                value = torch.randn(2, 4, 8, device="cuda")
                with accelerator.autocast():
                    loss = policy(value).float().square().mean()
                accelerator.backward(loss)
                optimizer.step()

                block = next(module for module in policy.modules() if isinstance(module, DtypeRecordingBlock))
                self.assertEqual(block.forward_dtypes, [torch.bfloat16, torch.bfloat16])
            finally:
                dist.destroy_process_group()
