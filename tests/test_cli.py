"""
测试 LLM 微调 CLI 工具中「不依赖 torch」的部分：
- cmd_init（创建目录结构 + 写示例 yaml，使用 torch-free 的 Config）
- main() 无子命令时打印帮助

train / eval / merge 三个命令需要 torch/transformers/peft，不在本环境测试。
"""

import argparse
import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
import cli  # noqa: E402


class TestCliTorchFree(unittest.TestCase):
    def _run_in_tmp(self, fn):
        d = tempfile.TemporaryDirectory()
        old = os.getcwd()
        os.chdir(d.name)
        try:
            return fn(d.name)
        finally:
            os.chdir(old)
            d.cleanup()

    def test_cmd_init_creates_structure(self):
        """cmd_init 应创建标准目录并写出默认配置 yaml"""

        def fn(_tmp):
            cli.cmd_init(argparse.Namespace())
            for sub in ["datasets", "checkpoints", "configs", "notebooks", "outputs"]:
                self.assertTrue(os.path.isdir(sub), f"缺少目录 {sub}")
            cfg = os.path.join("configs", "default.yaml")
            self.assertTrue(os.path.exists(cfg))
            # yaml 应可被 Config 解析
            from llm_finetuning import Config, FinetuningMethod

            c = Config.from_yaml(cfg)
            self.assertEqual(c.finetuning_method, FinetuningMethod.LORA)

        self._run_in_tmp(fn)

    def test_main_init_via_dispatch(self):
        """main(['init']) 应触发 cmd_init 创建目录"""

        def fn(_tmp):
            old_argv = sys.argv
            sys.argv = ["cli.py", "init"]
            try:
                cli.main()
            finally:
                sys.argv = old_argv
            self.assertTrue(os.path.isdir("datasets"))
            self.assertTrue(os.path.exists(os.path.join("configs", "default.yaml")))

        self._run_in_tmp(fn)

    def test_main_no_command_prints_help(self):
        """无子命令时 main() 应打印帮助而非报错"""
        old_argv = sys.argv
        sys.argv = ["cli.py"]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                cli.main()
        finally:
            sys.argv = old_argv
        out = buf.getvalue()
        self.assertIn("usage", out.lower())
        self.assertIn("init", out)

    def test_main_unknown_command_exits(self):
        """传入未知子命令时 argparse 应退出（非零码）"""
        old_argv = sys.argv
        sys.argv = ["cli.py", "definitely-not-a-command"]
        try:
            with self.assertRaises(SystemExit):
                cli.main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
