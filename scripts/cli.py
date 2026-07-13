#!/usr/bin/env python3
"""
LLM 微调 CLI 工具
命令行界面，快速开始微调
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_init(args):
    """初始化项目结构"""
    print("🚀 初始化 LLM 微调项目...")

    dirs = ["datasets", "checkpoints", "configs", "notebooks", "outputs"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"   ✅ 创建目录: {d}/")

    # 创建示例配置
    from llm_finetuning import Config, FinetuningMethod, LoRAConfig

    config = Config(
        model_name_or_path="bigscience/bloom-560m",
        finetuning_method=FinetuningMethod.LORA,
        output_dir="./checkpoints/default",
    )
    config.save_to_yaml("./configs/default.yaml")
    print(f"   ✅ 创建配置: configs/default.yaml")

    print("\n✅ 初始化完成！")


def cmd_train(args):
    """开始微调"""
    print(f"🎯 开始微调，使用配置: {args.config}")

    try:
        from llm_finetuning import Config, FinetuningEngine
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        sys.exit(1)

    # 加载配置
    if os.path.exists(args.config):
        config = Config.from_yaml(args.config)
    else:
        from llm_finetuning import FinetuningMethod
        config = Config(
            model_name_or_path=args.model or "bigscience/bloom-560m",
            finetuning_method=FinetuningMethod(args.method or "lora"),
            output_dir=args.output or "./checkpoints/cli",
        )

    # 覆盖参数
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size

    print(f"   模型: {config.model_name_or_path}")
    print(f"   方法: {config.finetuning_method.value}")
    print(f"   Epochs: {config.training.num_train_epochs}")
    print(f"   学习率: {config.training.learning_rate}")

    # 创建引擎
    engine = FinetuningEngine(config)

    # 加载数据
    if args.dataset:
        print(f"\n📦 加载数据集: {args.dataset}")
        train_dataset, eval_dataset = engine.load_data(
            dataset_name=args.dataset,
            dataset_path=args.data_path if args.data_path else None,
        )

        # 开始训练
        print("\n🚀 开始训练...")
        engine.train(train_dataset, eval_dataset)

        # 评估
        print("\n🔍 评估模型...")
        result = engine.evaluate(eval_dataset)
        print(f"✅ 最终困惑度: {result.perplexity:.2f}")

    else:
        print("\n⚠️  未指定数据集，仅完成模型加载")
        print("   使用 --dataset <name> 或 --data-path <path> 指定数据")


def cmd_eval(args):
    """评估模型"""
    print(f"🔍 评估模型: {args.model_path}")
    print(f"   数据集: {args.dataset}")
    # TODO: 实现模型评估逻辑
    # 需要加载模型、加载数据集、运行推理、计算指标
    print("   ⚠️ 此功能尚未实现 (TODO)")


def cmd_merge(args):
    """合并 LoRA 适配器"""
    print(f"🔄 合并模型适配器")
    print(f"   适配器路径: {args.adapter}")
    print(f"   输出路径: {args.output}")
    # TODO: 实现 LoRA 适配器合并逻辑
    # 需要加载基础模型、加载适配器、合并权重、保存模型
    print("   ⚠️ 此功能尚未实现 (TODO)")


def main():
    parser = argparse.ArgumentParser(
        description="LLM 微调 CLI 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 初始化项目
  python scripts/cli.py init

  # 使用配置文件训练
  python scripts/cli.py train --config configs/default.yaml --dataset imdb

  # 快速训练，命令行指定参数
  python scripts/cli.py train --model bigscience/bloom-560m --method lora --epochs 3

  # 使用本地数据
  python scripts/cli.py train --data-path ./datasets/my_data.jsonl
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # init 命令
    init_parser = subparsers.add_parser("init", help="初始化项目结构")

    # train 命令
    train_parser = subparsers.add_parser("train", help="开始模型微调")
    train_parser.add_argument("--config", type=str, help="配置文件路径")
    train_parser.add_argument("--model", type=str, help="模型名称或路径")
    train_parser.add_argument("--method", type=str, choices=["lora", "qlora", "full"],
                             help="微调方法")
    train_parser.add_argument("--dataset", type=str, help="HuggingFace 数据集名称")
    train_parser.add_argument("--data-path", type=str, help="本地数据集路径")
    train_parser.add_argument("--output", type=str, help="输出目录")
    train_parser.add_argument("--epochs", type=int, help="训练轮数")
    train_parser.add_argument("--lr", type=float, help="学习率")
    train_parser.add_argument("--batch-size", type=int, help="批次大小")

    # eval 命令
    eval_parser = subparsers.add_parser("eval", help="评估已训练模型")
    eval_parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    eval_parser.add_argument("--dataset", type=str, required=True, help="评估数据集")

    # merge 命令
    merge_parser = subparsers.add_parser("merge", help="合并 LoRA 适配器")
    merge_parser.add_argument("--adapter", type=str, required=True, help="适配器路径")
    merge_parser.add_argument("--base-model", type=str, help="基础模型")
    merge_parser.add_argument("--output", type=str, required=True, help="输出路径")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "merge":
        cmd_merge(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
