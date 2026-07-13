"""
训练可视化模块
支持损失曲线、学习率曲线、评估指标可视化
"""

import os
from typing import Any, Dict, List, Optional
from pathlib import Path


class TrainingVisualizer:
    """训练可视化工具"""

    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_loss_curve(
        self,
        train_losses: List[float],
        eval_losses: Optional[List[float]] = None,
        title: str = "训练损失曲线",
        save_path: Optional[str] = None,
    ) -> str:
        """绘制损失曲线"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            plt.figure(figsize=(10, 6))

            # 训练损失
            steps = np.arange(len(train_losses))
            plt.plot(steps, train_losses, label='训练损失', linewidth=2, color='#1f77b4')

            # 评估损失
            if eval_losses:
                eval_steps = np.linspace(0, len(train_losses) - 1, len(eval_losses))
                plt.plot(eval_steps, eval_losses, label='验证损失', linewidth=2,
                        color='#ff7f0e', marker='o', markersize=4)

            plt.xlabel('训练步数', fontsize=12)
            plt.ylabel('损失 (Loss)', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)

            # 保存
            if save_path is None:
                save_path = os.path.join(self.output_dir, 'loss_curve.png')

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ 损失曲线已保存到: {save_path}")
            return save_path

        except ImportError:
            print("⚠️  matplotlib 未安装，跳过绘图")
            return ""

    def plot_learning_rate_curve(
        self,
        learning_rates: List[float],
        title: str = "学习率变化曲线",
        save_path: Optional[str] = None,
    ) -> str:
        """绘制学习率曲线"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            plt.figure(figsize=(10, 6))

            steps = np.arange(len(learning_rates))
            plt.plot(steps, learning_rates, linewidth=2, color='#2ca02c')

            plt.xlabel('训练步数', fontsize=12)
            plt.ylabel('学习率', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

            if save_path is None:
                save_path = os.path.join(self.output_dir, 'learning_rate_curve.png')

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ 学习率曲线已保存到: {save_path}")
            return save_path

        except ImportError:
            print("⚠️  matplotlib 未安装，跳过绘图")
            return ""

    def plot_evaluation_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "模型评估对比",
        save_path: Optional[str] = None,
    ) -> str:
        """绘制多个模型/检查点的评估对比"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # 提取数据
            checkpoint_names = list(metrics.keys())
            metric_names = list(metrics[checkpoint_names[0]].keys())

            # 为每个指标创建子图
            n_metrics = len(metric_names)
            fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

            if n_metrics == 1:
                axes = [axes]

            x = np.arange(len(checkpoint_names))
            width = 0.6

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            for i, metric in enumerate(metric_names):
                values = [metrics[chk][metric] for chk in checkpoint_names]
                bars = axes[i].bar(x, values, width, color=colors[i % len(colors)])
                axes[i].set_xlabel('检查点', fontsize=11)
                axes[i].set_ylabel(metric, fontsize=11)
                axes[i].set_title(f'{metric} 对比', fontsize=12, fontweight='bold')
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(checkpoint_names, rotation=45, ha='right')
                axes[i].grid(True, alpha=0.3, axis='y')

                # 在柱子上显示数值
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{height:.2f}',
                        ha='center',
                        va='bottom',
                    )

            plt.tight_layout()

            if save_path is None:
                save_path = os.path.join(self.output_dir, 'evaluation_comparison.png')

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ 评估对比图已保存到: {save_path}")
            return save_path

        except ImportError:
            print("⚠️  matplotlib 未安装，跳过绘图")
            return ""

    def plot_perplexity_comparison(
        self,
        models: Dict[str, float],
        title: str = "模型困惑度对比",
        save_path: Optional[str] = None,
    ) -> str:
        """绘制多个模型的困惑度对比"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            plt.figure(figsize=(10, 6))

            model_names = list(models.keys())
            ppl_values = list(models.values())

            # 按困惑度排序（越低越好）
            sorted_indices = np.argsort(ppl_values)
            model_names = [model_names[i] for i in sorted_indices]
            ppl_values = [ppl_values[i] for i in sorted_indices]

            # 创建水平柱状图
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(model_names)))
            bars = plt.barh(model_names, ppl_values, color=colors)

            plt.xlabel('困惑度 (Perplexity) - 越低越好', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')

            # 在柱子上显示数值
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}',
                    ha='left',
                    va='center',
                )

            plt.gca().invert_yaxis()  # 最好的在上面

            if save_path is None:
                save_path = os.path.join(self.output_dir, 'perplexity_comparison.png')

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ 困惑度对比图已保存到: {save_path}")
            return save_path

        except ImportError:
            print("⚠️  matplotlib 未安装，跳过绘图")
            return ""

    def create_training_report(
        self,
        train_losses: List[float],
        eval_losses: List[float],
        learning_rates: List[float],
        final_metrics: Dict[str, float],
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> str:
        """创建完整的训练报告（HTML格式）"""
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_report.html')

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>训练报告 - {model_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a365d; border-bottom: 2px solid #3182ce; padding-bottom: 10px; }}
        h2 {{ color: #2d3748; font-size: 1.2em; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                       gap: 15px; }}
        .metric {{ background: #edf2f7; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-label {{ color: #4a5568; font-size: 0.9em; }}
        .metric-value {{ color: #2b6cb0; font-size: 1.5em; font-weight: bold; }}
        img {{ max-width: 100%; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>📊 训练报告: {model_name}</h1>
        <p>生成时间: {Path(save_path).stat().st_mtime if os.path.exists(save_path) else 'N/A'}</p>
    </div>

    <div class="card">
        <h2>📈 最终评估指标</h2>
        <div class="metric-grid">
"""

        # 添加指标卡片
        for metric_name, value in final_metrics.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            html_content += f"""
            <div class="metric">
                <div class="metric-label">{metric_name}</div>
                <div class="metric-value">{formatted_value}</div>
            </div>
"""

        html_content += f"""
        </div>
    </div>

    <div class="card">
        <h2>📉 训练曲线</h2>
        <img src="loss_curve.png" alt="损失曲线">
    </div>

    <div class="card">
        <h2>📈 学习率变化</h2>
        <img src="learning_rate_curve.png" alt="学习率曲线">
    </div>
</body>
</html>
"""

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ 训练报告已生成: {save_path}")
        return save_path
