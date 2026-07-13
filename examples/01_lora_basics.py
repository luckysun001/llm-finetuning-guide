"""
示例 1: 使用 LoRA 微调小型 GPT 模型
快速入门教程，演示完整的微调流程
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 60)
    print("📚 示例 1: LoRA 微调入门教程")
    print("=" * 60)

    try:
        from llm_finetuning import (
            Config,
            FinetuningMethod,
            FinetuningEngine,
            create_sample_instruction_dataset,
        )
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请先安装依赖: pip install torch transformers datasets peft trl evaluate")
        return

    # 1. 创建配置
    print("\n1️⃣  创建配置...")
    config = Config(
        model_name_or_path="bigscience/bloom-560m",  # 使用小模型演示
        finetuning_method=FinetuningMethod.LORA,
        output_dir="./checkpoints/lora_demo",
    )

    # 自定义训练参数
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 4
    config.training.gradient_accumulation_steps = 2
    config.training.learning_rate = 2e-4
    config.training.logging_steps = 5
    config.training.save_strategy = "no"  # 演示模式不保存检查点
    config.data.max_seq_length = 128

    print(f"   模型: {config.model_name_or_path}")
    print(f"   方法: {config.finetuning_method.value}")
    print(f"   学习率: {config.training.learning_rate}")

    # 2. 创建示例数据集
    print("\n2️⃣  创建示例数据集...")
    dataset_path = create_sample_instruction_dataset(
        output_path="./datasets/sample_instructions.jsonl",
        num_samples=50,
    )

    # 3. 创建微调引擎
    print("\n3️⃣  初始化微调引擎...")
    engine = FinetuningEngine(config)

    # 4. 加载数据
    print("\n4️⃣  加载数据集...")
    train_dataset, eval_dataset = engine.load_data(dataset_path=dataset_path)

    # 5. 定义指令分词函数
    def tokenize_instructions(examples):
        from llm_finetuning.data.processor import DataPreprocessor
        preprocessor = DataPreprocessor(engine.finetuner.tokenizer, config.data)
        return preprocessor.tokenize_instruction(examples)

    # 6. 开始训练
    print("\n5️⃣  开始训练 (演示模式，实际会训练)...")
    print("   注意: 完整训练可能需要几分钟，取决于硬件")

    # 实际运行训练（取消注释执行）
    # engine.train(train_dataset, eval_dataset, tokenize_func=tokenize_instructions)

    print("\n✅ 教程演示完成！")
    print("\n📝 下一步:")
    print("   1. 取消注释 engine.train() 行开始实际训练")
    print("   2. 查看 examples/02_qlora_demo.py 学习 QLoRA 4位微调")
    print("   3. 使用自己的数据集替换示例数据")
    print("   4. 阅读 README.md 了解更多功能")


if __name__ == "__main__":
    main()
