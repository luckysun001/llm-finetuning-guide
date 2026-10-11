"""YAML 配置解析测试

覆盖 training / data / evaluation / lora 四段的加载，
以及仓库内 configs/ 示例配置的可用性。
"""
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from llm_finetuning import Config, FinetuningMethod

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestFromYaml(unittest.TestCase):
    """从 YAML 加载"""

    def _write(self, data: dict) -> str:
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "cfg.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)
        return path

    def test_training_and_data_sections(self):
        path = self._write({
            "model_name_or_path": "m",
            "finetuning_method": "lora",
            "training": {"num_train_epochs": 7, "learning_rate": 1e-4},
            "data": {"max_seq_length": 1024},
        })
        cfg = Config.from_yaml(path)
        self.assertEqual(cfg.training.num_train_epochs, 7)
        self.assertEqual(cfg.training.learning_rate, 1e-4)
        self.assertEqual(cfg.data.max_seq_length, 1024)

    def test_evaluation_section(self):
        path = self._write({
            "model_name_or_path": "m",
            "evaluation": {"max_new_tokens": 64, "temperature": 0.3},
        })
        cfg = Config.from_yaml(path)
        self.assertEqual(cfg.evaluation.max_new_tokens, 64)
        self.assertEqual(cfg.evaluation.temperature, 0.3)

    def test_lora_section(self):
        path = self._write({
            "model_name_or_path": "m",
            "finetuning_method": "qlora",
            "lora": {"r": 32, "lora_alpha": 64, "target_modules": ["q_proj"]},
        })
        cfg = Config.from_yaml(path)
        self.assertEqual(cfg.lora_config.r, 32)
        self.assertEqual(cfg.lora_config.lora_alpha, 64)
        self.assertEqual(cfg.lora_config.target_modules, ["q_proj"])

    def test_lora_section_ignored_for_full_finetuning(self):
        """全参微调没有 lora_config，lora 段应被安全忽略"""
        path = self._write({
            "model_name_or_path": "m",
            "finetuning_method": "full",
            "lora": {"r": 32},
        })
        cfg = Config.from_yaml(path)
        self.assertFalse(hasattr(cfg, "lora_config"))

    def test_unknown_keys_are_ignored(self):
        """配置文件里出现未知字段不应报错"""
        path = self._write({
            "model_name_or_path": "m",
            "training": {"not_a_real_field": 1},
            "unknown_section": {"x": 1},
        })
        cfg = Config.from_yaml(path)
        self.assertEqual(cfg.model_name_or_path, "m")

    def test_round_trip(self):
        """保存后重新加载应保持关键字段一致"""
        cfg = Config(
            model_name_or_path="round-trip",
            finetuning_method=FinetuningMethod.QLORA,
        )
        cfg.training.num_train_epochs = 4
        cfg.lora_config.r = 16

        path = os.path.join(tempfile.mkdtemp(), "rt.yaml")
        cfg.save_to_yaml(path)
        loaded = Config.from_yaml(path)

        self.assertEqual(loaded.training.num_train_epochs, 4)
        self.assertEqual(loaded.lora_config.r, 16)


class TestRepoExampleConfigs(unittest.TestCase):
    """仓库自带的示例配置应当可被解析"""

    def test_examples_load(self):
        for name in ("lora_qwen.yaml", "qlora_7b.yaml"):
            path = REPO_ROOT / "configs" / name
            self.assertTrue(path.exists(), f"缺少示例配置: {name}")

            cfg = Config.from_yaml(str(path))
            self.assertTrue(cfg.model_name_or_path)
            self.assertIn(cfg.finetuning_method, list(FinetuningMethod))


if __name__ == "__main__":
    unittest.main()
