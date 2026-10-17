"""
超参数搜索示例
==============

演示如何用 ``llm_finetuning.tuning`` 搜索 LoRA 微调的超参。

为了在没有 GPU 的环境也能直接运行，这里的 ``objective`` 是一个**代理指标**：
假设"学习率越接近 2e-4、LoRA 秩越大越好"，用公式模拟一次训练的效果。

换成真实训练时，只需要把 objective 换成真正跑训练并返回验证 loss 的函数，
其余代码完全不变::

    def objective(params):
        config = Config("Qwen/Qwen2.5-7B-Instruct", finetuning_method=FinetuningMethod.LORA)
        apply_params(config, params)
        trainer = FinetuningEngine(config).train(dataset)
        return trainer.evaluate()["eval_loss"]

运行::

    python examples/04_hyperparameter_search.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_finetuning.config import Config, FinetuningMethod  # noqa: E402
from llm_finetuning.tuning import (  # noqa: E402
    HyperparameterSearcher,
    apply_params,
    default_lora_space,
)


def objective(params: dict) -> float:
    """代理目标函数：越小越好

    真实场景请替换为"跑一次训练并返回验证 loss / 困惑度"。
    """
    lr_penalty = abs(math.log(params["learning_rate"] / 2e-4))
    # 秩越大拟合能力越强，但收益递减
    rank_gain = math.log(params["lora.r"]) / 10.0
    # batch size 太小会不稳定
    batch_penalty = 0.15 if params["training.per_device_train_batch_size"] == 2 else 0.0
    # warmup 多一点更稳
    warmup_gain = min(params["training.warmup_steps"], 100) / 1000.0
    return lr_penalty - rank_gain + batch_penalty - warmup_gain


def main() -> None:
    space = default_lora_space()
    print(space.describe())
    print(f"网格搜索总组合数: {space.grid_size()}\n")

    # ---------- 1. 随机搜索（便宜，适合大空间） ----------
    searcher = HyperparameterSearcher(
        space,
        objective,
        direction="minimize",
        seed=42,
        on_trial_end=lambda t: print(f"  trial #{t.trial_id}: score={t.score:.4f} {t.params}"),
    )
    print("随机搜索 12 次：")
    random_result = searcher.random_search(n_trials=12)
    print(random_result.summary())

    # ---------- 2. 网格搜索（贵，但覆盖均匀） ----------
    print("\n网格搜索（每个连续参数取 3 个采样点，最多 20 次）：")
    grid_searcher = HyperparameterSearcher(space, objective, direction="minimize", max_trials=20)
    grid_result = grid_searcher.grid_search(max_points=3)
    print(grid_result.summary())

    print("\n最好的 3 组参数：")
    for i, trial in enumerate(grid_result.top_k(3), start=1):
        print(f"  {i}. score={trial.score:.4f}  {trial.params}")

    # ---------- 3. 把最优参数写回配置 ----------
    config = Config("Qwen/Qwen2.5-7B-Instruct", finetuning_method=FinetuningMethod.LORA)
    applied = apply_params(config, grid_result.best_params)
    print(f"\n已写入配置的字段: {applied}")
    print(f"  learning_rate = {config.training.learning_rate}")
    print(f"  lora.r        = {config.lora_config.r}")
    print(f"  batch_size    = {config.training.per_device_train_batch_size}")

    # ---------- 4. 结果落盘 ----------
    os.makedirs("outputs", exist_ok=True)
    grid_result.save_csv("outputs/hparam_search.csv")
    grid_result.save_json("outputs/hparam_search.json")
    print("\n结果已保存到 outputs/hparam_search.csv 与 .json")


if __name__ == "__main__":
    main()
