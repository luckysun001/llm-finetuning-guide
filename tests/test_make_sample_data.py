"""
测试示例数据生成脚本（纯 Python，无 torch 依赖）。
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# 将仓库根目录下的 scripts/ 加入 sys.path，以便直接 import 脚本模块
_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, os.path.abspath(_SCRIPTS_DIR))

import make_sample_data  # noqa: E402


class TestMakeSampleData(unittest.TestCase):
    def test_samples_well_formed(self):
        """内置样本应包含 instruction/input/output 字段"""
        self.assertGreater(len(make_sample_data.SAMPLES), 0)
        for item in make_sample_data.SAMPLES:
            self.assertIn("instruction", item)
            self.assertIn("input", item)
            self.assertIn("output", item)

    def test_write_jsonl(self):
        """write_jsonl 应写出等量的 jsonl 行"""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "data.jsonl"
            count = make_sample_data.write_jsonl(make_sample_data.SAMPLES, out)
            self.assertEqual(count, len(make_sample_data.SAMPLES))
            self.assertTrue(out.exists())
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), len(make_sample_data.SAMPLES))
            # 每行都是合法 JSON 且字段齐全
            for line in lines:
                obj = json.loads(line)
                self.assertIn("instruction", obj)

    def test_main_writes_default_count(self):
        """main() 应写出 SAMPLES 数量的样本"""
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "sample_instructions.jsonl")
            old_argv = sys.argv
            sys.argv = ["make_sample_data.py", "--output", out]
            try:
                make_sample_data.main()
            finally:
                sys.argv = old_argv
            self.assertTrue(os.path.exists(out))
            with open(out, "r", encoding="utf-8") as f:
                n = sum(1 for _ in f)
            self.assertEqual(n, len(make_sample_data.SAMPLES))


if __name__ == "__main__":
    unittest.main()
