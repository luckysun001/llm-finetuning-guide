"""
数据处理模块测试 - 多格式加载和数据预处理
运行方法: python -m pytest tests/test_processor.py -v
"""

import unittest
import tempfile
import json
import os

import pandas as pd

try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


class TestDatasetLoaderJSON(unittest.TestCase):
    """JSON 数据加载测试"""

    def test_load_from_json(self):
        """应正确加载 JSON 文件"""
        from llm_finetuning.data.processor import DatasetLoader
        from llm_finetuning import DataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            data = [
                {"instruction": "hello", "output": "hi"},
                {"instruction": "bye", "output": "goodbye"},
            ]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f)

            loader = DatasetLoader(DataConfig())
            result = loader.load_from_json(filepath)

            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["instruction"], "hello")
            self.assertEqual(result[1]["output"], "goodbye")

    def test_load_from_json_empty(self):
        """加载空 JSON 数组应返回空列表"""
        from llm_finetuning.data.processor import DatasetLoader
        from llm_finetuning import DataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "empty.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([], f)

            loader = DatasetLoader(DataConfig())
            result = loader.load_from_json(filepath)
            self.assertEqual(len(result), 0)


class TestDatasetLoaderJSONL(unittest.TestCase):
    """JSONL 数据加载测试"""

    def test_load_from_jsonl(self):
        """应正确加载 JSONL 文件"""
        from llm_finetuning.data.processor import DatasetLoader
        from llm_finetuning import DataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.jsonl")
            lines = [
                {"instruction": "q1", "output": "a1"},
                {"instruction": "q2", "output": "a2"},
                {"instruction": "q3", "output": "a3"},
            ]
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in lines:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            loader = DatasetLoader(DataConfig())
            result = loader.load_from_jsonl(filepath)

            self.assertEqual(len(result), 3)
            self.assertEqual(result[0]["instruction"], "q1")
            self.assertEqual(result[2]["output"], "a3")

    def test_load_from_jsonl_with_empty_lines(self):
        """JSONL 中的空行应被跳过"""
        from llm_finetuning.data.processor import DatasetLoader
        from llm_finetuning import DataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.jsonl")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json.dumps({"a": 1}) + "\n")
                f.write("\n")
                f.write("   \n")
                f.write(json.dumps({"a": 2}) + "\n")

            loader = DatasetLoader(DataConfig())
            result = loader.load_from_jsonl(filepath)
            self.assertEqual(len(result), 2)


class TestDatasetLoaderCSV(unittest.TestCase):
    """CSV 数据加载测试"""

    def test_load_from_csv(self):
        """应正确加载 CSV 文件"""
        from llm_finetuning.data.processor import DatasetLoader
        from llm_finetuning import DataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.csv")
            df = pd.DataFrame({
                "instruction": ["q1", "q2"],
                "output": ["a1", "a2"],
            })
            df.to_csv(filepath, index=False)

            loader = DatasetLoader(DataConfig())
            result = loader.load_from_csv(filepath)

            self.assertEqual(len(result), 2)
            self.assertIn("instruction", result.columns)
            self.assertEqual(result.iloc[0]["output"], "a1")


class TestDatasetLoaderParquet(unittest.TestCase):
    """Parquet 数据加载测试"""

    @unittest.skipUnless(_HAS_PYARROW, "pyarrow not installed")
    def test_load_from_parquet(self):
        """应正确加载 Parquet 文件"""
        from llm_finetuning.data.processor import DatasetLoader
        from llm_finetuning import DataConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.parquet")
            df = pd.DataFrame({
                "text": ["hello", "world"],
                "label": [0, 1],
            })
            df.to_parquet(filepath)

            loader = DatasetLoader(DataConfig())
            result = loader.load_from_parquet(filepath)

            self.assertEqual(len(result), 2)
            self.assertIn("text", result.columns)


class TestDatasetLoaderHub(unittest.TestCase):
    """HuggingFace Hub 加载测试"""

    def test_load_from_hub_no_name_raises(self):
        """未指定数据集名应抛出 ValueError"""
        from llm_finetuning.data.processor import DatasetLoader
        from llm_finetuning import DataConfig

        loader = DatasetLoader(DataConfig())
        with self.assertRaises(ValueError):
            loader.load_from_hub(None)


class TestCreateSampleDataset(unittest.TestCase):
    """示例数据集创建测试"""

    def test_create_sample_dataset_content(self):
        """创建的示例数据集应包含有效字段"""
        from llm_finetuning.data.processor import create_sample_instruction_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "sample.jsonl")
            result_path = create_sample_instruction_dataset(
                output_path=output_path,
                num_samples=5,
            )

            with open(result_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.assertIn("instruction", data)
                    self.assertIn("input", data)
                    self.assertIn("output", data)
                    self.assertIsInstance(data["instruction"], str)
                    self.assertGreater(len(data["instruction"]), 0)

    def test_create_sample_dataset_different_sizes(self):
        """不同数量应生成不同行数"""
        from llm_finetuning.data.processor import create_sample_instruction_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            path_5 = os.path.join(tmpdir, "s5.jsonl")
            path_20 = os.path.join(tmpdir, "s20.jsonl")

            create_sample_instruction_dataset(output_path=path_5, num_samples=5)
            create_sample_instruction_dataset(output_path=path_20, num_samples=20)

            with open(path_5, 'r') as f:
                count_5 = sum(1 for _ in f)
            with open(path_20, 'r') as f:
                count_20 = sum(1 for _ in f)

            self.assertEqual(count_5, 5)
            self.assertEqual(count_20, 20)


class TestDataConfigEdgeCases(unittest.TestCase):
    """DataConfig 边界测试"""

    def test_custom_data_config(self):
        """自定义配置值应正确设置"""
        from llm_finetuning import DataConfig

        config = DataConfig(
            max_seq_length=1024,
            text_column="content",
            test_size=0.2,
            shuffle=False,
        )
        self.assertEqual(config.max_seq_length, 1024)
        self.assertEqual(config.text_column, "content")
        self.assertEqual(config.test_size, 0.2)
        self.assertFalse(config.shuffle)

    def test_data_config_with_dataset_name(self):
        """设置 dataset_name 应正确存储"""
        from llm_finetuning import DataConfig

        config = DataConfig(dataset_name="databricks/databricks-dolly-15k")
        self.assertEqual(config.dataset_name, "databricks/databricks-dolly-15k")


if __name__ == "__main__":
    unittest.main()
