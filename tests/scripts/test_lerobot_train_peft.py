import unittest
from types import SimpleNamespace

from lerobot.scripts.lerobot_train import _is_pi05_full_parameter_finetune


class Pi05PeftClassificationTest(unittest.TestCase):
    def test_peft_is_not_classified_as_full_parameter_training(self) -> None:
        policy = SimpleNamespace(type="pi05", train_expert_only=False, freeze_vision_encoder=False)
        full_finetune_cfg = SimpleNamespace(policy=policy, peft=None)
        peft_cfg = SimpleNamespace(policy=policy, peft=SimpleNamespace(method_type="LORA"))

        self.assertTrue(_is_pi05_full_parameter_finetune(full_finetune_cfg))
        self.assertFalse(_is_pi05_full_parameter_finetune(peft_cfg))


if __name__ == "__main__":
    unittest.main()
