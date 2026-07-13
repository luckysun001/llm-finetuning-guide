"""
LLM 微调学习项目
================

一个面向初学者的大模型微调开源学习项目，
包含多种微调方法（LoRA, QLoRA 等）、
数据处理、训练、评估、可视化的完整实现。

快速开始:
    >>> from llm_finetuning import Config, FinetuningEngine, quick_finetune

核心模块:
    - config: 配置管理
    - engine: 训练引擎
    - methods: 微调方法实现 (LoRA, QLoRA)
    - data: 数据处理
    - evaluation: 模型评估
    - visualization: 训练可视化

许可证: MIT
"""

__version__ = "1.0.0"
__author__ = "AI Learning Lab"
__license__ = "MIT"

# 核心配置
from llm_finetuning.config import (
    Config,
    FinetuningMethod,
    QuantizationLevel,
    ModelType,
    LoRAConfig,
    QLoRAConfig,
    TrainingConfig,
    DataConfig,
    EvaluationConfig,
)

# 微调方法
from llm_finetuning.methods.lora import (
    LoRAFinetuner,
    QLoRAFinetuner,
)

# 数据处理
from llm_finetuning.data.processor import (
    DatasetLoader,
    DataPreprocessor,
    DatasetSplitter,
    create_sample_instruction_dataset,
)

# 评估
from llm_finetuning.evaluation.evaluator import (
    ModelEvaluator,
    EvaluationResult,
    ComparisonEvaluator,
)

# 可视化
from llm_finetuning.visualization.plotter import (
    TrainingVisualizer,
)

# 训练引擎
from llm_finetuning.engine import (
    FinetuningEngine,
    TrainingCallback,
    quick_finetune,
)

__all__ = [
    # 配置
    "Config",
    "FinetuningMethod",
    "QuantizationLevel",
    "ModelType",
    "LoRAConfig",
    "QLoRAConfig",
    "TrainingConfig",
    "DataConfig",
    "EvaluationConfig",
    # 微调方法
    "LoRAFinetuner",
    "QLoRAFinetuner",
    # 数据处理
    "DatasetLoader",
    "DataPreprocessor",
    "DatasetSplitter",
    "create_sample_instruction_dataset",
    # 评估
    "ModelEvaluator",
    "EvaluationResult",
    "ComparisonEvaluator",
    # 可视化
    "TrainingVisualizer",
    # 引擎
    "FinetuningEngine",
    "TrainingCallback",
    "quick_finetune",
]
