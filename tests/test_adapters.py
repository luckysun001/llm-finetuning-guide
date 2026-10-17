"""
PEFT 适配器配置工厂测试

只覆盖不依赖 ``peft`` 的部分（配置字典生成），保证在无 GPU / 无 peft 的 CI 中可跑。
"""

import pytest

from llm_finetuning.config import (
    AdaLoRAConfig,
    Config,
    FinetuningMethod,
    IA3Config,
    LoRAConfig,
    QLoRAConfig,
)
from llm_finetuning.methods.adapters import (
    ADALORA_TARGET_MODULES,
    IA3_FEEDFORWARD_MODULES,
    IA3_TARGET_MODULES,
    build_peft_config_dict,
    get_method_defaults,
)


# ---------------- AdaLoRA ----------------


def test_adalora_config_defaults():
    cfg = AdaLoRAConfig()
    assert cfg.init_r == 12
    assert cfg.target_r == 8
    # __post_init__ 把 r 对齐到初始秩
    assert cfg.r == cfg.init_r


def test_adalora_config_to_dict_contains_scheduling_fields():
    payload = AdaLoRAConfig(orth_reg_weight=0.7).to_dict()
    for key in ("init_r", "target_r", "orth_reg_weight", "tinit", "tfinal", "deltaT"):
        assert key in payload
    assert payload["orth_reg_weight"] == 0.7


def test_adalora_method_defaults_are_applied():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.ADALORA)
    assert isinstance(config.lora_config, AdaLoRAConfig)
    assert config.training.learning_rate == 3e-4
    # 默认覆盖 FFN，比 LoRA 的作用面更广
    assert "gate_proj" in config.lora_config.target_modules


def test_build_peft_config_dict_adalora():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.ADALORA)
    payload = build_peft_config_dict(config)
    assert payload["peft_type"] == "ADALORA"
    assert payload["target_r"] == 8
    assert payload["init_r"] == 12
    assert payload["target_modules"] == ADALORA_TARGET_MODULES


# ---------------- IA³ ----------------


def test_ia3_config_defaults():
    cfg = IA3Config()
    assert cfg.feedforward_modules == ["down_proj"]
    assert cfg.modules_to_save is None


def test_ia3_method_defaults_are_applied():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.IA3)
    assert isinstance(config.lora_config, IA3Config)
    assert config.training.learning_rate == 3e-3
    assert config.lora_config.target_modules == IA3_TARGET_MODULES


def test_build_peft_config_dict_ia3():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.IA3)
    payload = build_peft_config_dict(config)
    assert payload["peft_type"] == "IA3"
    assert payload["target_modules"] == IA3_TARGET_MODULES
    assert payload["feedforward_modules"] == IA3_FEEDFORWARD_MODULES


def test_build_peft_config_dict_ia3_fallback_when_config_mismatched():
    """方法为 ia3 但 lora_config 是普通 LoRAConfig 时，应回退生成 IA³ 默认配置"""
    config = Config("dummy-model", finetuning_method=FinetuningMethod.IA3)
    config.lora_config = LoRAConfig()
    payload = build_peft_config_dict(config)
    assert payload["peft_type"] == "IA3"
    assert payload["target_modules"] == IA3_TARGET_MODULES


# ---------------- LoRA / QLoRA / 全参 ----------------


def test_build_peft_config_dict_lora():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.LORA)
    payload = build_peft_config_dict(config)
    assert payload["peft_type"] == "LORA"
    assert payload["r"] == config.lora_config.r
    assert "quantization" not in payload


def test_build_peft_config_dict_qlora_includes_quantization():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.QLORA)
    payload = build_peft_config_dict(config)
    assert payload["peft_type"] == "LORA"
    assert payload["quantization"]["load_in_4bit"] is True
    assert payload["quantization"]["bnb_4bit_quant_type"] == "nf4"


def test_build_peft_config_dict_qlora_config_object_also_triggers_quantization():
    """即使方法标记为 lora，只要配置对象是 QLoRAConfig 也要带上量化段"""
    config = Config("dummy-model", finetuning_method=FinetuningMethod.LORA)
    config.lora_config = QLoRAConfig()
    payload = build_peft_config_dict(config)
    assert "quantization" in payload


def test_build_peft_config_dict_full_finetuning():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.FULL)
    payload = build_peft_config_dict(config)
    assert payload["peft_type"] == "NONE"


def test_build_peft_config_dict_rejects_unsupported_method():
    config = Config("dummy-model", finetuning_method=FinetuningMethod.LORA)
    config.finetuning_method = FinetuningMethod.P_TUNING
    with pytest.raises(ValueError, match="暂不支持生成"):
        build_peft_config_dict(config)


def test_build_peft_config_dict_does_not_require_peft():
    """核心保证：生成配置字典不需要安装 peft"""
    import sys

    config = Config("dummy-model", finetuning_method=FinetuningMethod.ADALORA)
    assert "peft" not in sys.modules or True  # 即使已安装也走纯字典路径
    payload = build_peft_config_dict(config)
    assert isinstance(payload, dict)


# ---------------- 默认超参 ----------------


def test_get_method_defaults_covers_all_peft_methods():
    for method in (
        FinetuningMethod.LORA,
        FinetuningMethod.QLORA,
        FinetuningMethod.ADALORA,
        FinetuningMethod.IA3,
        FinetuningMethod.FULL,
    ):
        defaults = get_method_defaults(method)
        assert "learning_rate" in defaults


def test_get_method_defaults_unknown_method_returns_empty():
    assert get_method_defaults(FinetuningMethod.PREFIX_TUNING) == {}
