"""
测试 ``python -m llm_finetuning.deploy`` 入口（不依赖 torch）。

- build_parser() 返回带默认值的解析器
- 缺 --model-path 时 argparse 退出
- 缺少 torch/transformers 等依赖时，run_server 抛 ImportError，main() 捕获并返回 1
  （不会真的去绑定端口启动服务）
"""

import sys
import unittest
from unittest import mock

import llm_finetuning.deploy.__main__ as dm


class TestDeployMain(unittest.TestCase):
    def test_build_parser_defaults(self):
        parser = dm.build_parser()
        ns = parser.parse_args(["--model-path", "/tmp/model"])
        self.assertEqual(ns.model_path, "/tmp/model")
        self.assertEqual(ns.backend, "transformers")
        self.assertEqual(ns.host, "0.0.0.0")
        self.assertEqual(ns.port, 8000)
        self.assertEqual(ns.device, "auto")
        self.assertEqual(ns.dtype, "auto")

    def test_build_parser_choices(self):
        parser = dm.build_parser()
        ns = parser.parse_args(
            ["--model-path", "/m", "--backend", "vllm", "--port", "9000", "--device", "cpu"]
        )
        self.assertEqual(ns.backend, "vllm")
        self.assertEqual(ns.port, 9000)
        self.assertEqual(ns.device, "cpu")

    def test_main_missing_model_path_exits(self):
        """缺少必填的 --model-path 时 argparse 应退出"""
        with self.assertRaises(SystemExit):
            dm.main([])

    def test_main_no_torch_returns_1(self):
        """缺少 torch 等依赖、run_server 抛 ImportError 时，main 捕获并返回 1（不崩溃）"""
        with mock.patch("llm_finetuning.deploy.run_server", side_effect=ImportError("no torch")):
            rc = dm.main(["--model-path", "/nonexistent/model", "--backend", "transformers"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
