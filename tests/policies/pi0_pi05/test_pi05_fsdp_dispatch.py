#!/usr/bin/env python

import os
import socket
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import checkpoint

from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel


class TinyPaliGemmaWithExpert(PaliGemmaWithExpertModel):
    """Tiny parameter paths exercising the production dispatch without loading PI0.5 weights."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.patch_embedding = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.token_embedding = nn.Embedding(16, 4)
        self.prefix_proj = nn.Linear(4, 4)
        self.expert_proj = nn.Linear(4, 4)

    def embed_image(self, image, *, frame_mask=None, temporal_attention_every=4):
        del frame_mask, temporal_attention_every
        return self.patch_embedding(image).flatten(2).transpose(1, 2)

    def embed_language_tokens(self, tokens):
        return self.token_embedding(tokens)

    def _forward_joint(self, *, inputs_embeds=None, **kwargs):
        del kwargs
        prefix, suffix = inputs_embeds
        return [self.prefix_proj(prefix), self.expert_proj(suffix)], None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestPI05FSDPDispatch(unittest.TestCase):
    def test_dispatch_preserves_unsharded_numerics(self) -> None:
        torch.manual_seed(0)
        model = TinyPaliGemmaWithExpert()
        image = torch.randn(1, 3, 8, 8)
        tokens = torch.tensor([[1, 2, 3]])
        suffix = torch.randn(1, 3, 4)

        torch.testing.assert_close(
            model(mode="embed_image", image=image),
            model.embed_image(image),
        )
        language_embeds = model(mode="embed_language_tokens", tokens=tokens)
        torch.testing.assert_close(language_embeds, model.embed_language_tokens(tokens))
        actual = model(mode="joint", inputs_embeds=[language_embeds, suffix])
        expected = model._forward_joint(inputs_embeds=[language_embeds, suffix])
        torch.testing.assert_close(actual[0][0], expected[0][0])
        torch.testing.assert_close(actual[0][1], expected[0][1])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FSDP2")
    def test_unshards_ordinary_tensor_inputs(self) -> None:
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
                model = TinyPaliGemmaWithExpert().cuda()
                fully_shard(model, reshard_after_forward=True)

                image = torch.randn(1, 3, 8, 8, device="cuda")
                tokens = torch.tensor([[1, 2, 3]], device="cuda")
                suffix = torch.randn(1, 3, 4, device="cuda")

                image_embeds = checkpoint(
                    lambda value: model(mode="embed_image", image=value),
                    image,
                    use_reentrant=False,
                )
                language_embeds = checkpoint(
                    lambda value: model(mode="embed_language_tokens", tokens=value),
                    tokens,
                    use_reentrant=False,
                )

                def joint_forward(prefix, expert_suffix):
                    (prefix_out, suffix_out), _ = model(
                        mode="joint", inputs_embeds=[prefix, expert_suffix]
                    )
                    return prefix_out, suffix_out

                prefix_out, suffix_out = checkpoint(
                    joint_forward,
                    language_embeds,
                    suffix,
                    use_reentrant=False,
                )

                self.assertEqual(image_embeds.shape, (1, 16, 4))
                self.assertEqual(prefix_out.shape, suffix_out.shape)
                self.assertEqual(prefix_out.shape, (1, 3, 4))
                (image_embeds.sum() + prefix_out.sum() + suffix_out.sum()).backward()
            finally:
                dist.destroy_process_group()
