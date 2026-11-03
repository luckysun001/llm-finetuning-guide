"""
测试训练可视化模块。

绘图函数内部惰性 `import matplotlib.pyplot`，本测试在导入前将后端设为 Agg，
使其在无 GUI 的 CI / 本地环境也能正常生成图片文件。
"""

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 导入前设置

from llm_finetuning.visualization.plotter import TrainingVisualizer


class TestTrainingVisualizer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = os.path.join(self.tmp.name, "viz")
        self.viz = TrainingVisualizer(self.out_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_creates_output_dir(self):
        """构造器应创建输出目录"""
        self.assertTrue(os.path.isdir(self.out_dir))

    def test_plot_loss_curve_saves_png(self):
        """损失曲线应生成非空 png 并返回路径"""
        path = self.viz.plot_loss_curve([1.0, 0.7, 0.4, 0.2])
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)
        self.assertTrue(path.endswith("loss_curve.png"))

    def test_plot_loss_curve_with_eval(self):
        """带验证损失时仍能生成图片"""
        path = self.viz.plot_loss_curve(
            [1.0, 0.7, 0.4], eval_losses=[1.2, 0.9, 0.6], save_path="eval.png"
        )
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_learning_rate_curve(self):
        """学习率曲线应生成 png"""
        path = self.viz.plot_learning_rate_curve([1e-4, 1e-4, 5e-5])
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)
        self.assertTrue(path.endswith("learning_rate_curve.png"))

    def test_plot_evaluation_comparison(self):
        """多检查点评估对比图应生成 png"""
        metrics = {
            "ckpt_a": {"accuracy": 0.81, "f1": 0.78},
            "ckpt_b": {"accuracy": 0.85, "f1": 0.82},
        }
        path = self.viz.plot_evaluation_comparison(metrics)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_perplexity_comparison(self):
        """困惑度对比图应生成 png"""
        models = {"model_a": 12.3, "model_b": 9.1, "model_c": 15.7}
        path = self.viz.plot_perplexity_comparison(models)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_create_training_report_html(self):
        """训练报告应写出 HTML 且包含模型名与指标"""
        report_path = self.viz.create_training_report(
            train_losses=[1.0, 0.5],
            eval_losses=[1.1, 0.6],
            learning_rates=[1e-4, 5e-5],
            final_metrics={"accuracy": 0.88, "perplexity": 9.2},
            model_name="test-model",
        )
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("test-model", content)
        self.assertIn("accuracy", content)
        self.assertIn("0.8800", content)  # 0.88 格式化为 4 位小数


if __name__ == "__main__":
    unittest.main()
