"""
测试数据处理模块
"""

import unittest
import tempfile
import json
import os


class TestDatasetLoader(unittest.TestCase):
    """数据集加载器测试"""

    def test_create_sample_dataset(self):
        """测试创建示例数据集"""
        from llm_finetuning.data.processor import create_sample_instruction_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_instructions.jsonl")

            # 创建数据集
            result_path = create_sample_instruction_dataset(
                output_path=output_path,
                num_samples=10,
            )

            # 验证文件存在
            self.assertTrue(os.path.exists(result_path))

            # 验证内容
            with open(result_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 10)

                # 验证每条都是有效的 JSON
                for line in lines:
                    data = json.loads(line.strip())
                    self.assertIn("instruction", data)
                    self.assertIn("input", data)
                    self.assertIn("output", data)


class TestDataConfig(unittest.TestCase):
    """数据配置测试"""

    def test_data_config_defaults(self):
        from llm_finetuning import DataConfig

        config = DataConfig()
        self.assertEqual(config.max_seq_length, 512)
        self.assertEqual(config.text_column, "text")
        self.assertEqual(config.test_size, 0.1)
        self.assertTrue(config.shuffle)

    def test_evaluation_config_defaults(self):
        from llm_finetuning import EvaluationConfig

        config = EvaluationConfig()
        self.assertIn("perplexity", config.metrics)
        self.assertEqual(config.eval_batch_size, 8)


class TestEvaluationResult(unittest.TestCase):
    """评估结果测试"""

    def test_evaluation_result_creation(self):
        from llm_finetuning.evaluation.evaluator import EvaluationResult

        result = EvaluationResult(perplexity=15.5, bleu_score=0.75)

        self.assertEqual(result.perplexity, 15.5)
        self.assertEqual(result.bleu_score, 0.75)
        self.assertIsInstance(result.rouge_scores, dict)

    def test_evaluation_result_to_dict(self):
        from llm_finetuning.evaluation.evaluator import EvaluationResult

        result = EvaluationResult(perplexity=15.5)
        result_dict = result.to_dict()

        self.assertIn("perplexity", result_dict)
        self.assertEqual(result_dict["perplexity"], 15.5)

    def test_evaluation_result_repr(self):
        from llm_finetuning.evaluation.evaluator import EvaluationResult

        result = EvaluationResult(perplexity=15.5)
        repr_str = repr(result)

        self.assertIn("PPL=15.50", repr_str)


if __name__ == "__main__":
    unittest.main()
