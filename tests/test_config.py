"""
测试配置模块
"""

import unittest
import tempfile
import os


class TestConfig(unittest.TestCase):
    """配置类测试"""

    def test_finetuning_method_enum(self):
        """测试微调方法枚举"""
        from llm_finetuning import FinetuningMethod

        self.assertEqual(FinetuningMethod.LORA.value, "lora")
        self.assertEqual(FinetuningMethod.QLORA.value, "qlora")
        self.assertEqual(FinetuningMethod.FULL.value, "full")

    def test_quantization_level_enum(self):
        """测试量化级别枚举"""
        from llm_finetuning import QuantizationLevel

        self.assertEqual(QuantizationLevel.FP32.value, "fp32")
        self.assertEqual(QuantizationLevel.INT4.value, "int4")

    def test_config_creation(self):
        """测试配置创建"""
        from llm_finetuning import Config, FinetuningMethod

        config = Config(
            model_name_or_path="test-model",
            finetuning_method=FinetuningMethod.LORA,
            output_dir="./test-output",
        )

        self.assertEqual(config.model_name_or_path, "test-model")
        self.assertEqual(config.finetuning_method, FinetuningMethod.LORA)
        self.assertEqual(config.training.output_dir, "./test-output")

    def test_lora_config_defaults(self):
        """测试 LoRA 配置默认值"""
        from llm_finetuning import LoRAConfig

        lora_config = LoRAConfig()
        self.assertEqual(lora_config.r, 8)
        self.assertEqual(lora_config.lora_alpha, 16)
        self.assertEqual(lora_config.lora_dropout, 0.05)

    def test_qlora_config_defaults(self):
        """测试 QLoRA 配置默认值"""
        from llm_finetuning import QLoRAConfig

        qlora_config = QLoRAConfig()
        self.assertTrue(qlora_config.load_in_4bit)
        self.assertTrue(qlora_config.bnb_4bit_use_double_quant)
        self.assertEqual(qlora_config.bnb_4bit_quant_type, "nf4")

    def test_training_config_defaults(self):
        """测试训练配置默认值"""
        from llm_finetuning import TrainingConfig

        train_config = TrainingConfig()
        self.assertEqual(train_config.num_train_epochs, 3)
        self.assertEqual(train_config.per_device_train_batch_size, 8)
        self.assertEqual(train_config.learning_rate, 2e-5)  # TrainingConfig 默认值; LoRA/QLoRA 会覆写为 2e-4

    def test_config_yaml_save_load(self):
        """测试配置 YAML 保存和加载"""
        from llm_finetuning import Config, FinetuningMethod

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.yaml")

            # 创建并保存配置
            config1 = Config(
                model_name_or_path="test-model",
                finetuning_method=FinetuningMethod.LORA,
                output_dir="./test",
            )
            config1.training.num_train_epochs = 5
            config1.save_to_yaml(config_path)

            # 加载配置
            config2 = Config.from_yaml(config_path)

            # 验证
            self.assertEqual(config2.model_name_or_path, "test-model")
            self.assertEqual(config2.training.num_train_epochs, 5)


if __name__ == "__main__":
    unittest.main()
