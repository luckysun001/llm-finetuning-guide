"""
PEFT 适配器配置工厂：LoRA / QLoRA / AdaLoRA / IA³

设计要点：
- :func:`build_peft_config_dict` **不依赖 peft**，返回纯字典，方便测试与配置落盘
- :func:`build_peft_config` 才真正构造 ``peft`` 的配置对象，缺失依赖时给出安装提示
- 重型依赖一律函数内按需导入，保持 ``poetry install`` 轻量

配置数据类（``AdaLoRAConfig`` / ``IA3Config``）定义在 :mod:`llm_finetuning.config`，
与 ``LoRAConfig`` / ``QLoRAConfig`` 保持一致。
"""

from typing import Any, Dict, List

from llm_finetuning.config import (
    Config,
    FinetuningMethod,
    LoRAConfig,
    QLoRAConfig,
    AdaLoRAConfig,
    IA3Config,
)

__all__ = [
    "ADALORA_TARGET_MODULES",
    "IA3_TARGET_MODULES",
    "IA3_FEEDFORWARD_MODULES",
    "build_peft_config_dict",
    "build_peft_config",
    "get_method_defaults",
]

# AdaLoRA 默认作用模块：相比 LoRA 额外覆盖 FFN，因为秩分配需要更广的搜索面
ADALORA_TARGET_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# IA³ 只作用于注意力与 FFN 的关键向量，参数量极小（远低于 LoRA）
IA3_TARGET_MODULES: List[str] = ["q_proj", "v_proj", "down_proj"]
IA3_FEEDFORWARD_MODULES: List[str] = ["down_proj"]


def get_method_defaults(method: FinetuningMethod) -> Dict[str, Any]:
    """返回该微调方法的推荐默认超参（用于展示与配置生成）"""
    if method == FinetuningMethod.LORA:
        return {"learning_rate": 2e-4, "r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
    if method == FinetuningMethod.QLORA:
        return {
            "learning_rate": 2e-4,
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "quantization_level": "int4",
        }
    if method == FinetuningMethod.ADALORA:
        return {
            "learning_rate": 3e-4,
            "init_r": 12,
            "target_r": 8,
            "lora_alpha": 32,
            "orth_reg_weight": 0.5,
        }
    if method == FinetuningMethod.IA3:
        return {"learning_rate": 3e-3, "target_modules": list(IA3_TARGET_MODULES)}
    if method == FinetuningMethod.FULL:
        return {"learning_rate": 5e-6, "gradient_checkpointing": True}
    return {}


def build_peft_config_dict(config: Config) -> Dict[str, Any]:
    """根据 :class:`Config` 生成对应的 PEFT 配置字典（不依赖 peft）

    返回值可直接写入 YAML 或打印，便于在没有 GPU / 没有 peft 的环境下
    先确认配置是否符合预期。
    """
    method = config.finetuning_method
    payload: Dict[str, Any] = {
        "method": method.value if hasattr(method, "value") else str(method),
        "task_type": getattr(getattr(config, "lora_config", None), "task_type", "CAUSAL_LM"),
    }

    if method in (FinetuningMethod.LORA, FinetuningMethod.QLORA):
        lora: Any = getattr(config, "lora_config", LoRAConfig())
        payload.update(
            {
                "peft_type": "LORA",
                "r": lora.r,
                "lora_alpha": lora.lora_alpha,
                "target_modules": list(lora.target_modules),
                "lora_dropout": lora.lora_dropout,
                "bias": lora.bias,
            }
        )
        if method == FinetuningMethod.QLORA or isinstance(lora, QLoRAConfig):
            payload["quantization"] = {
                "load_in_4bit": getattr(lora, "load_in_4bit", True),
                "bnb_4bit_use_double_quant": getattr(lora, "bnb_4bit_use_double_quant", True),
                "bnb_4bit_quant_type": getattr(lora, "bnb_4bit_quant_type", "nf4"),
                "bnb_4bit_compute_dtype": getattr(lora, "bnb_4bit_compute_dtype", "bfloat16"),
            }
        return payload

    if method == FinetuningMethod.ADALORA:
        cfg: Any = getattr(config, "lora_config", None)
        if not isinstance(cfg, AdaLoRAConfig):
            cfg = AdaLoRAConfig(
                target_modules=list(ADALORA_TARGET_MODULES),
                total_step=config.training.max_steps if config.training.max_steps > 0 else None,
            )
        payload.update({"peft_type": "ADALORA", **cfg.to_dict()})
        return payload

    if method == FinetuningMethod.IA3:
        cfg = getattr(config, "lora_config", None)
        if not isinstance(cfg, IA3Config):
            cfg = IA3Config(
                target_modules=list(IA3_TARGET_MODULES),
                feedforward_modules=list(IA3_FEEDFORWARD_MODULES),
            )
        payload.update({"peft_type": "IA3", **cfg.to_dict()})
        return payload

    if method == FinetuningMethod.FULL:
        payload["peft_type"] = "NONE"
        payload["note"] = "全参数微调不使用 PEFT 适配器"
        return payload

    raise ValueError(
        f"暂不支持生成 {method} 的 PEFT 配置"
        f"（已支持: lora / qlora / adalora / ia3 / full）"
    )


def build_peft_config(config: Config) -> Any:
    """构造真正的 ``peft`` 配置对象

    Raises:
        ImportError: 未安装 ``peft`` 时给出安装提示
        ValueError: 微调方法不使用 PEFT 适配器时
    """
    try:
        from peft import AdaLoraConfig, IA3Config as PeftIA3Config, LoraConfig, TaskType
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise ImportError(
            "peft 未安装，无法构造 PEFT 配置对象，请执行: pip install peft\n"
            "（若只想查看配置字典，可改用 build_peft_config_dict，它不依赖 peft）"
        ) from exc

    payload = build_peft_config_dict(config)
    peft_type = payload["peft_type"]
    task_type = getattr(TaskType, payload.get("task_type", "CAUSAL_LM"), TaskType.CAUSAL_LM)

    if peft_type == "LORA":
        return LoraConfig(
            r=payload["r"],
            lora_alpha=payload["lora_alpha"],
            target_modules=payload["target_modules"],
            lora_dropout=payload["lora_dropout"],
            bias=payload["bias"],
            task_type=task_type,
        )

    if peft_type == "ADALORA":
        return AdaLoraConfig(
            init_r=payload["init_r"],
            target_r=payload["target_r"],
            lora_alpha=payload["lora_alpha"],
            target_modules=payload["target_modules"],
            lora_dropout=payload["lora_dropout"],
            bias=payload["bias"],
            orth_reg_weight=payload["orth_reg_weight"],
            tinit=payload["tinit"],
            tfinal=payload["tfinal"],
            deltaT=payload["deltaT"],
            total_step=payload["total_step"],
            task_type=task_type,
        )

    if peft_type == "IA3":
        return PeftIA3Config(
            target_modules=payload["target_modules"],
            feedforward_modules=payload["feedforward_modules"],
            modules_to_save=payload["modules_to_save"],
            task_type=task_type,
        )

    raise ValueError(f"{config.finetuning_method} 不使用 PEFT 适配器，无需构造配置")
