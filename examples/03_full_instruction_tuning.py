"""
示例 3: 完整的指令微调流程
使用真实的指令数据集进行微调
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_alpaca_style_dataset(output_path: str, num_samples: int = 200):
    """创建 Alpaca 风格的指令数据集"""
    import random

    # 指令模板
    instruction_templates = [
        "解释以下概念：{}",
        "用简单的语言解释什么是{}",
        "列出{}的三个主要特点",
        "比较{}和{}的区别",
        "写一段关于{}的介绍",
        "说明{}的工作原理",
        "给出{}的实际应用场景",
        "总结{}的优缺点",
        "如何学习{}？给出建议",
        "{}的未来发展趋势是什么？",
    ]

    # 主题
    topics = [
        "人工智能", "机器学习", "深度学习", "神经网络", "自然语言处理",
        "计算机视觉", "大语言模型", "Transformer", "注意力机制", "词嵌入",
        "梯度下降", "反向传播", "激活函数", "损失函数", "优化器",
        "正则化", "过拟合", "欠拟合", "特征工程", "数据增强",
    ]

    samples = []
    for i in range(num_samples):
        topic1 = random.choice(topics)
        topic2 = random.choice([t for t in topics if t != topic1])
        instruction = random.choice(instruction_templates).format(topic1, topic2)

        # 生成简单的输出
        output = f"这是关于「{topic1}」的回答。\n\n"
        output += f"{topic1}是人工智能领域的重要概念，主要具有以下特点：\n"
        output += "1. 能够处理复杂的模式识别任务\n"
        output += "2. 具有强大的泛化能力\n"
        output += "3. 可以从数据中自动学习特征\n\n"
        output += f"在实际应用中，{topic1}被广泛应用于各个领域。"

        samples.append({
            "instruction": instruction,
            "input": "",
            "output": output,
        })

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✅ 已创建 {num_samples} 条指令数据: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("🎯 示例 3: 完整指令微调流程")
    print("=" * 70)

    try:
        from llm_finetuning import (
            Config,
            FinetuningMethod,
            LoRAConfig,
            FinetuningEngine,
        )
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请先安装依赖")
        return

    # 1. 创建数据集
    print("\n1️⃣  创建指令数据集...")
    dataset_path = create_alpaca_style_dataset(
        output_path="./datasets/alpaca_style_instructions.jsonl",
        num_samples=200,
    )

    # 2. 创建配置
    print("\n2️⃣  创建微调配置...")
    config = Config(
        model_name_or_path="bigscience/bloom-560m",
        finetuning_method=FinetuningMethod.LORA,
        output_dir="./checkpoints/instruction_tuning",
    )

    # LoRA 配置
    config.lora_config = LoRAConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
    )

    # 训练配置
    config.training.num_train_epochs = 2
    config.training.per_device_train_batch_size = 4
    config.training.gradient_accumulation_steps = 4
    config.training.learning_rate = 3e-4
    config.training.warmup_steps = 50
    config.training.logging_steps = 10
    config.training.save_strategy = "epoch"
    config.training.evaluation_strategy = "epoch"
    config.training.load_best_model_at_end = True

    # 数据配置
    config.data.max_seq_length = 256
    config.data.text_column = "text"

    print(f"   模型: {config.model_name_or_path}")
    print(f"   LoRA 秩: {config.lora_config.r}")
    print(f"   学习率: {config.training.learning_rate}")
    print(f"   Epochs: {config.training.num_train_epochs}")

    # 3. 保存配置文件
    config_path = "./configs/instruction_tuning.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config.save_to_yaml(config_path)
    print(f"\n3️⃣  配置已保存: {config_path}")

    # 4. 初始化引擎
    print("\n4️⃣  初始化微调引擎...")
    engine = FinetuningEngine(config)

    # 5. 加载数据
    print("\n5️⃣  加载数据集...")
    train_dataset, eval_dataset = engine.load_data(dataset_path=dataset_path)

    # 6. 定义指令格式化函数
    def format_instruction(examples):
        """Alpaca 风格的指令格式化"""
        prompts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples["input"][i] if "input" in examples else ""
            output = examples["output"][i]

            if input_text:
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
            else:
                prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
            prompts.append(prompt)

        # 分词
        tokenized = engine.finetuner.tokenizer(
            prompts,
            truncation=True,
            max_length=config.data.max_seq_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("\n6️⃣  数据格式化函数已定义")

    print("\n" + "=" * 70)
    print("✅ 完整微调流程准备完成！")
    print("=" * 70)
    print("\n🚀 执行以下代码开始训练：")
    print("""
    # 开始训练
    engine.train(
        train_dataset,
        eval_dataset,
        tokenize_func=format_instruction,
    )

    # 评估模型
    result = engine.evaluate(eval_dataset)
    print(f"最终困惑度: {result.perplexity:.2f}")

    # 合并并保存完整模型
    merged_path = engine.merge_and_save()
""")
    print(f"\n📁 输出目录: {config.training.output_dir}")
    print(f"⚙️  配置文件: {config_path}")


if __name__ == "__main__":
    main()
