from draccus import parse

from lerobot.configs.default import PeftConfig


def test_peft_config_parses_per_module_rank_and_alpha_patterns() -> None:
    cfg = parse(
        PeftConfig,
        args=[
            "--rank_pattern",
            '{".*gemma_expert\\\\..*":32}',
            "--alpha_pattern",
            '{".*gemma_expert\\\\..*":32}',
        ],
    )

    assert cfg.r == 16
    assert cfg.rank_pattern == {r".*gemma_expert\..*": 32}
    assert cfg.alpha_pattern == {r".*gemma_expert\..*": 32}
