"""
大模型微调项目配置管理
"""

import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict


class FinetuningMethod(str, Enum):
    """微调方法枚举"""
    FULL = "full"                    # 全参数微调
    LORA = "lora"                    # Low-Rank Adaptation
    QLORA = "qlora"                  # Quantized LoRA (4-bit)
    ADALORA = "adalora"              # Adaptive LoRA
    IA3 = "ia3"                      # Infused Adapter by Inhibiting and Amplifying Inner Activations
    P_TUNING = "p_tuning"            # P-Tuning
    PREFIX_TUNING = "prefix_tuning"  # Prefix Tuning
    PROMPT_TUNING = "prompt_tuning"  # Prompt Tuning


class QuantizationLevel(str, Enum):
    """量化级别"""
    FP32 = "fp32"    # 32位浮点数
    FP16 = "fp16"    # 16位浮点数
    BF16 = "bf16"    # Brain Floating Point 16
    INT8 = "int8"    # 8位整数
    INT4 = "int4"    # 4位整数


class ModelType(str, Enum):
    """模型类型"""
    CAUSAL_LM = "causal_lm"          # 因果语言模型 (GPT, LLaMA)
    SEQ2SEQ = "seq2seq"              # 序列到序列 (T5, BART)
    CLASSIFICATION = "classification"  # 分类模型


@dataclass
class LoRAConfig:
    """LoRA 配置"""
    r: int = 8                        # LoRA 秩
    lora_alpha: int = 16              # LoRA alpha 参数
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05       # Dropout
    bias: str = "none"                # bias 类型
    task_type: str = "CAUSAL_LM"     # 任务类型

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QLoRAConfig(LoRAConfig):
    """QLoRA 配置"""
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"  # 归一化浮点数
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = -1

    # 精度设置
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False

    # 优化器和调度器
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"

    # 日志和保存
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 3

    # 其他
    seed: int = 42
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = ""
    dataset_path: Optional[str] = None
    text_column: str = "text"
    label_column: Optional[str] = None
    max_seq_length: int = 512
    train_split: str = "train"
    eval_split: str = "test"
    test_size: float = 0.1
    shuffle: bool = True
    seed: int = 42

    # 数据处理
    truncation: bool = True
    padding: str = "max_length"
    add_special_tokens: bool = True


@dataclass
class EvaluationConfig:
    """评估配置"""
    metrics: List[str] = field(default_factory=lambda: ["perplexity"])
    eval_batch_size: int = 8
    max_eval_samples: Optional[int] = None
    generate_during_training: bool = True
    num_beams: int = 4
    max_new_tokens: int = 128
    temperature: float = 1.0


class Config:
    """全局配置"""

    def __init__(
        self,
        model_name_or_path: str,
        finetuning_method: FinetuningMethod = FinetuningMethod.LORA,
        output_dir: str = "./checkpoints",
    ):
        # 模型配置
        self.model_name_or_path = model_name_or_path
        self.model_type = ModelType.CAUSAL_LM
        self.finetuning_method = finetuning_method
        self.quantization_level = QuantizationLevel.FP16

        # 各种子配置
        self.training = TrainingConfig(output_dir=output_dir)
        self.data = DataConfig()
        self.evaluation = EvaluationConfig()

        # 根据微调方法设置默认配置
        self._setup_method_defaults()

    def _setup_method_defaults(self):
        """根据微调方法设置默认配置"""
        if self.finetuning_method == FinetuningMethod.LORA:
            self.lora_config = LoRAConfig()
            self.training.learning_rate = 2e-4

        elif self.finetuning_method == FinetuningMethod.QLORA:
            self.lora_config = QLoRAConfig()
            self.quantization_level = QuantizationLevel.INT4
            self.training.learning_rate = 2e-4
            self.training.bf16 = True

        elif self.finetuning_method == FinetuningMethod.FULL:
            self.training.learning_rate = 5e-6
            self.training.gradient_checkpointing = True

    def save_to_yaml(self, filepath: str):
        """保存配置到 YAML 文件"""
        import yaml

        config_dict = {
            "model_name_or_path": self.model_name_or_path,
            "model_type": self.model_type.value,
            "finetuning_method": self.finetuning_method.value,
            "quantization_level": self.quantization_level.value,
            "training": self.training.to_dict(),
            "data": asdict(self.data),
            "evaluation": asdict(self.evaluation),
        }

        if hasattr(self, 'lora_config'):
            config_dict["lora"] = self.lora_config.to_dict()

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """从 YAML 文件加载配置"""
        import yaml

        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        config = cls(
            model_name_or_path=config_dict["model_name_or_path"],
            finetuning_method=FinetuningMethod(config_dict.get("finetuning_method", "lora")),
        )

        # 更新训练配置
        if "training" in config_dict:
            for k, v in config_dict["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        # 更新数据配置
        if "data" in config_dict:
            for k, v in config_dict["data"].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)

        # 更新评估配置
        if "evaluation" in config_dict:
            for k, v in config_dict["evaluation"].items():
                if hasattr(config.evaluation, k):
                    setattr(config.evaluation, k, v)

        # 更新 LoRA / QLoRA 配置（仅在该方法下存在 lora_config 时生效）
        if "lora" in config_dict and hasattr(config, "lora_config"):
            for k, v in config_dict["lora"].items():
                if hasattr(config.lora_config, k):
                    setattr(config.lora_config, k, v)

        return config

    def __repr__(self) -> str:
        return (
            f"Config(model={self.model_name_or_path}, "
            f"method={self.finetuning_method.value}, "
            f"lr={self.training.learning_rate})"
        )
