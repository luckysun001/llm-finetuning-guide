"""
示例 2: QLoRA 4位量化微调
在消费级显卡上微调大模型的首选方案
显著降低显存需求 (7B 模型只需 ~10GB 显存)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 60)
    print("⚡ 示例 2: QLoRA 4位量化微调教程")
    print("=" * 60)

    try:
        from llm_finetuning import (
            Config,
            FinetuningMethod,
            QLoRAConfig,
            FinetuningEngine,
            create_sample_instruction_dataset,
        )
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请先安装依赖: pip install torch transformers datasets peft bitsandbytes trl")
        return

    # 1. 创建 QLoRA 配置
    print("\n1️⃣  创建 QLoRA 配置...")
    config = Config(
        model_name_or_path="bigscience/bloom-1b7",  # 更大的模型
        finetuning_method=FinetuningMethod.QLORA,
        output_dir="./checkpoints/qlora_demo",
    )

    # 自定义 QLoRA 参数
    config.lora_config = QLoRAConfig(
        r=64,                          # 更高的秩 = 更强的表达能力
        lora_alpha=128,
        target_modules=["query_key_value"],  # Bloom 模型的注意力模块
        lora_dropout=0.05,
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",     # 归一化浮点数，效果更好
        bnb_4bit_compute_dtype="bfloat16",
    )

    # 训练参数
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 4
    config.training.gradient_accumulation_steps = 4
    config.training.learning_rate = 2e-4
    config.training.bf16 = True  # 使用 bfloat16 加速
    config.training.gradient_checkpointing = True  # 节省更多显存
    config.data.max_seq_length = 256

    print(f"   模型: {config.model_name_or_path}")
    print(f"   方法: {config.finetuning_method.value} (4位量化)")
    print(f"   LoRA 秩: {config.lora_config.r}")
    print(f"   量化类型: {config.lora_config.bnb_4bit_quant_type}")

    # 2. 创建示例数据集
    print("\n2️⃣  创建示例数据集...")
    dataset_path = create_sample_instruction_dataset(
        output_path="./datasets/qlora_sample.jsonl",
        num_samples=100,
    )

    # 3. 创建微调引擎
    print("\n3️⃣  初始化 QLoRA 微调引擎...")
    engine = FinetuningEngine(config)

    # 4. 加载数据
    print("\n4️⃣  加载数据集...")
    train_dataset, eval_dataset = engine.load_data(dataset_path=dataset_path)

    print("\n✅ QLoRA 配置准备完成！")
    print("\n💡 QLoRA 优势:")
    print("   - 7B 模型只需 ~10GB 显存即可微调")
    print("   - 13B 模型只需 ~15GB 显存")
    print("   - 70B 模型只需 ~40GB 显存")
    print("   - 性能几乎与全参数微调相当")
    print("\n📊 显存对比 (7B 模型):")
    print("   - 全参数微调 (FP16): ~120GB+")
    print("   - LoRA (FP16):       ~40GB")
    print("   - QLoRA (4-bit):    ~10GB ✨")
    print("\n⚠️   注意:")
    print("   - bitsandbytes 需要 NVIDIA 显卡")
    print("   - 确保 CUDA 版本兼容")


if __name__ == "__main__":
    main()
