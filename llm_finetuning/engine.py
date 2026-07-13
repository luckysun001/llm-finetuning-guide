"""
训练引擎 - 整合所有模块的核心入口
"""

import os
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path

from llm_finetuning.config import Config, FinetuningMethod
from llm_finetuning.methods.lora import LoRAFinetuner, QLoRAFinetuner
from llm_finetuning.data.processor import DatasetLoader, DataPreprocessor, DatasetSplitter
from llm_finetuning.evaluation.evaluator import ModelEvaluator
from llm_finetuning.visualization.plotter import TrainingVisualizer


class TrainingCallback:
    """训练回调基类"""

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """训练开始时调用"""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """训练结束时调用"""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Epoch 开始时调用"""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Epoch 结束时调用"""
        pass

    def on_step_begin(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """训练步开始时调用"""
        pass

    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """训练步结束时调用"""
        pass


class FinetuningEngine:
    """
    微调引擎 - 一站式大模型微调
    整合数据加载、预处理、训练、评估、可视化
    """

    def __init__(self, config: Config):
        self.config = config
        self.finetuner: Optional[Any] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.visualizer = TrainingVisualizer(
            os.path.join(config.training.output_dir, "visualizations")
        )
        self.callbacks: List[TrainingCallback] = []

        # 训练历史
        self.train_history: Dict[str, List[float]] = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
        }

        print(f"🚀 初始化微调引擎")
        print(f"   模型: {config.model_name_or_path}")
        print(f"   方法: {config.finetuning_method.value}")
        print(f"   输出目录: {config.training.output_dir}")

    def add_callback(self, callback: TrainingCallback) -> None:
        """添加训练回调"""
        self.callbacks.append(callback)

    def load_data(
        self,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        custom_tokenize_func: Optional[Callable] = None,
    ) -> tuple:
        """加载和预处理数据"""
        print("\n📦 数据加载阶段")
        print("=" * 50)

        # 1. 加载数据集
        loader = DatasetLoader(self.config.data)

        if dataset_path:
            if dataset_path.endswith('.jsonl'):
                from datasets import Dataset
                data = loader.load_from_jsonl(dataset_path)
                dataset = Dataset.from_list(data)
            elif dataset_path.endswith('.json'):
                from datasets import Dataset
                data = loader.load_from_json(dataset_path)
                dataset = Dataset.from_list(data)
            elif dataset_path.endswith('.csv'):
                df = loader.load_from_csv(dataset_path)
                from datasets import Dataset
                dataset = Dataset.from_pandas(df)
            else:
                raise ValueError(f"不支持的文件格式: {dataset_path}")
        else:
            dataset = loader.load_from_hub(dataset_name)

        # 2. 分割数据集
        if isinstance(dataset, dict):
            train_dataset = dataset.get(self.config.data.train_split)
            eval_dataset = dataset.get(self.config.data.eval_split)
            if eval_dataset is None and train_dataset is not None:
                train_dataset, eval_dataset = DatasetSplitter.split_train_test(
                    train_dataset,
                    test_size=self.config.data.test_size,
                    seed=self.config.data.seed,
                )
        else:
            train_dataset, eval_dataset = DatasetSplitter.split_train_test(
                dataset,
                test_size=self.config.data.test_size,
                seed=self.config.data.seed,
            )

        print(f"✅ 数据集准备完成")
        print(f"   训练集: {len(train_dataset)} 条")
        print(f"   验证集: {len(eval_dataset)} 条")

        return train_dataset, eval_dataset

    def prepare_tokenized_data(
        self,
        train_dataset,
        eval_dataset=None,
        tokenize_func: Optional[Callable] = None,
    ) -> tuple:
        """准备分词后的数据"""
        print("\n🔧 数据预处理阶段")
        print("=" * 50)

        # 确保模型已加载
        if self.finetuner is None or self.finetuner.tokenizer is None:
            self._setup_finetuner()

        preprocessor = DataPreprocessor(
            self.finetuner.tokenizer,
            self.config.data,
        )

        # 预处理训练集
        tokenized_train = preprocessor.apply(
            train_dataset,
            tokenize_func,
        )

        # 预处理验证集
        tokenized_eval = None
        if eval_dataset:
            tokenized_eval = preprocessor.apply(
                eval_dataset,
                tokenize_func,
            )

        return tokenized_train, tokenized_eval

    def _setup_finetuner(self) -> None:
        """设置微调器"""
        method = self.config.finetuning_method

        if method in [FinetuningMethod.LORA, FinetuningMethod.ADALORA, FinetuningMethod.IA3]:
            self.finetuner = LoRAFinetuner(self.config)
        elif method == FinetuningMethod.QLORA:
            self.finetuner = QLoRAFinetuner(self.config)
        elif method == FinetuningMethod.FULL:
            # 全参数微调可以使用基础的 Trainer
            self.finetuner = LoRAFinetuner(self.config)
        else:
            raise ValueError(f"不支持的微调方法: {method}")

        # 加载模型
        self.finetuner.load_model_and_tokenizer()

        # 应用 LoRA (如果需要)
        if method != FinetuningMethod.FULL:
            self.finetuner.apply_lora()

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        tokenize_func: Optional[Callable] = None,
    ) -> Any:
        """执行完整的训练流程"""
        # 1. 预处理数据
        tokenized_train, tokenized_eval = self.prepare_tokenized_data(
            train_dataset,
            eval_dataset,
            tokenize_func,
        )

        # 2. 开始训练
        print("\n🎯 训练阶段")
        print("=" * 50)

        # 触发回调
        for callback in self.callbacks:
            callback.on_train_begin()

        # 执行训练
        trainer = self.finetuner.train(tokenized_train, tokenized_eval)

        # 保存模型
        self.finetuner.save_model()

        # 触发回调
        for callback in self.callbacks:
            callback.on_train_end()

        print("\n✅ 训练完成！")
        return trainer

    def evaluate(
        self,
        eval_dataset,
        tokenize_func: Optional[Callable] = None,
        save_results: bool = True,
    ) -> Any:
        """评估模型"""
        print("\n🔍 模型评估阶段")
        print("=" * 50)

        # 确保数据已分词
        if self.finetuner is None or self.finetuner.tokenizer is None:
            self._setup_finetuner()

        # 预处理评估数据
        if not hasattr(eval_dataset[0], "input_ids"):
            preprocessor = DataPreprocessor(
                self.finetuner.tokenizer,
                self.config.data,
            )
            eval_dataset = preprocessor.apply(eval_dataset, tokenize_func)

        # 创建评估器
        self.evaluator = ModelEvaluator(
            self.finetuner.model,
            self.finetuner.tokenizer,
        )

        # 执行评估
        result = self.evaluator.evaluate(
            eval_dataset,
            compute_ppl=True,
            generate_samples=True,
        )

        # 保存结果
        if save_results:
            result_path = os.path.join(
                self.config.training.output_dir,
                "evaluation_results.json",
            )
            self.evaluator.save_results(result, result_path)

        # 生成可视化
        self._generate_visualizations(result)

        return result

    def _generate_visualizations(self, eval_result: Any) -> None:
        """生成训练可视化"""
        # 如果有训练历史，绘制损失曲线
        if self.train_history["train_loss"]:
            self.visualizer.plot_loss_curve(
                self.train_history["train_loss"],
                self.train_history.get("eval_loss"),
            )

        # 如果有学习率历史
        if self.train_history["learning_rate"]:
            self.visualizer.plot_learning_rate_curve(
                self.train_history["learning_rate"],
            )

        # 生成训练报告
        final_metrics = {
            "困惑度 (Perplexity)": eval_result.perplexity,
        }
        if hasattr(eval_result, 'bleu_score') and eval_result.bleu_score:
            final_metrics["BLEU"] = eval_result.bleu_score

        self.visualizer.create_training_report(
            self.train_history["train_loss"],
            self.train_history.get("eval_loss", []),
            self.train_history["learning_rate"],
            final_metrics,
            model_name=self.config.model_name_or_path,
        )

    def run_full_pipeline(
        self,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        tokenize_func: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        运行完整的微调流程
        数据加载 → 预处理 → 训练 → 评估 → 可视化
        """
        print("\n" + "=" * 70)
        print("🎯 LLM 微调完整流程")
        print("=" * 70)

        # 1. 加载数据
        train_dataset, eval_dataset = self.load_data(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )

        # 2. 训练
        trainer = self.train(train_dataset, eval_dataset, tokenize_func)

        # 3. 评估
        eval_result = self.evaluate(eval_dataset, tokenize_func)

        print("\n" + "=" * 70)
        print("🎉 所有流程完成！")
        print("=" * 70)
        print(f"📁 输出目录: {self.config.training.output_dir}")
        print(f"📊 最终困惑度: {eval_result.perplexity:.2f}")
        print("=" * 70)

        return {
            "trainer": trainer,
            "evaluation": eval_result,
            "output_dir": self.config.training.output_dir,
        }

    def merge_and_save(self, output_dir: Optional[str] = None) -> str:
        """合并适配器并保存完整模型"""
        if output_dir is None:
            output_dir = os.path.join(
                self.config.training.output_dir,
                "merged_model",
            )

        print(f"\n🔄 合并模型适配器...")
        self.finetuner.merge_and_save(output_dir)

        return output_dir


# ============================================
# 便捷函数
# ============================================

def quick_finetune(
    model_name: str,
    dataset_name: str,
    method: str = "lora",
    output_dir: str = "./checkpoints",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 8,
) -> FinetuningEngine:
    """
    快速微调 - 一键开始微调

    参数:
        model_name: 模型名称或路径
        dataset_name: 数据集名称
        method: 微调方法 (lora, qlora, full)
        output_dir: 输出目录
        num_epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 批次大小

    示例:
        >>> engine = quick_finetune(
        ...     model_name="bigscience/bloom-560m",
        ...     dataset_name="imdb",
        ...     method="lora",
        ... )
    """
    # 创建配置
    config = Config(
        model_name_or_path=model_name,
        finetuning_method=FinetuningMethod(method),
        output_dir=output_dir,
    )

    # 更新训练参数
    config.training.num_train_epochs = num_epochs
    config.training.learning_rate = learning_rate
    config.training.per_device_train_batch_size = batch_size

    # 创建引擎
    engine = FinetuningEngine(config)

    # 运行完整流程
    engine.run_full_pipeline(dataset_name=dataset_name)

    return engine
