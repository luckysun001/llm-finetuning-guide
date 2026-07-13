"""
数据处理模块
支持多种数据集格式和预处理方法
"""

import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from llm_finetuning.config import DataConfig


class DatasetLoader:
    """数据集加载器"""

    def __init__(self, config: DataConfig):
        self.config = config

    def load_from_hub(self, dataset_name: Optional[str] = None) -> Any:
        """从 HuggingFace Hub 加载数据集"""
        name = dataset_name or self.config.dataset_name
        if not name:
            raise ValueError("dataset_name 不能为空")

        try:
            from datasets import load_dataset

            print(f"📥 从 HuggingFace Hub 加载数据集: {name}")
            dataset = load_dataset(name)
            print(f"✅ 数据集加载完成: {dataset}")
            return dataset

        except ImportError:
            raise ImportError("datasets 模块未安装: pip install datasets")

    def load_from_json(self, filepath: str) -> List[Dict[str, Any]]:
        """从 JSON 文件加载数据"""
        print(f"📄 从 JSON 加载数据: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 加载了 {len(data)} 条数据")
        return data

    def load_from_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """从 JSONL 文件加载数据"""
        print(f"📄 从 JSONL 加载数据: {filepath}")
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"✅ 加载了 {len(data)} 条数据")
        return data

    def load_from_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """从 CSV 文件加载数据"""
        print(f"📄 从 CSV 加载数据: {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        print(f"✅ 加载了 {len(df)} 行数据")
        return df

    def load_from_parquet(self, filepath: str, **kwargs) -> pd.DataFrame:
        """从 Parquet 文件加载数据"""
        print(f"📄 从 Parquet 加载数据: {filepath}")
        df = pd.read_parquet(filepath, **kwargs)
        print(f"✅ 加载了 {len(df)} 行数据")
        return df


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, tokenizer: Any, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config

    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """基础分词函数"""
        text_column = self.config.text_column

        if isinstance(examples[text_column], str):
            texts = [examples[text_column]]
        else:
            texts = examples[text_column]

        tokenized = self.tokenizer(
            texts,
            truncation=self.config.truncation,
            padding=self.config.padding,
            max_length=self.config.max_seq_length,
            return_special_tokens_mask=True,
        )

        # 设置 labels（对于因果语言模型，labels = input_ids）
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def tokenize_instruction(
        self,
        examples: Dict[str, Any],
        instruction_column: str = "instruction",
        input_column: str = "input",
        output_column: str = "output",
    ) -> Dict[str, Any]:
        """
        指令微调格式的分词
        适用于 Alpaca/Guanaco 等指令数据集格式
        """
        prompts = []

        for i in range(len(examples[instruction_column])):
            instruction = examples[instruction_column][i]
            input_text = examples.get(input_column, [""])[i]
            output_text = examples[output_column][i]

            if input_text:
                prompt = f"""### 指令:
{instruction}

### 输入:
{input_text}

### 输出:
{output_text}"""
            else:
                prompt = f"""### 指令:
{instruction}

### 输出:
{output_text}"""

            prompts.append(prompt)

        tokenized = self.tokenizer(
            prompts,
            truncation=self.config.truncation,
            padding=self.config.padding,
            max_length=self.config.max_seq_length,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def tokenize_chat(
        self,
        examples: Dict[str, Any],
        conversation_column: str = "conversations",
    ) -> Dict[str, Any]:
        """
        对话格式的分词
        适用于 ShareGPT 等多轮对话数据集
        """
        formatted_conversations = []

        for conv in examples[conversation_column]:
            # 格式化对话
            formatted = self._format_conversation(conv)
            formatted_conversations.append(formatted)

        tokenized = self.tokenizer(
            formatted_conversations,
            truncation=self.config.truncation,
            padding=self.config.padding,
            max_length=self.config.max_seq_length,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """格式化多轮对话"""
        formatted = ""
        for turn in conversation:
            role = turn.get("role", turn.get("from", "user"))
            content = turn.get("content", turn.get("value", ""))

            if role.lower() in ["user", "human", "h"]:
                formatted += f"### 用户:\n{content}\n\n"
            elif role.lower() in ["assistant", "gpt", "bot"]:
                formatted += f"### 助手:\n{content}\n\n"
            elif role.lower() in ["system", "sys"]:
                formatted += f"### 系统:\n{content}\n\n"

        return formatted

    def apply(
        self,
        dataset: Any,
        tokenize_func: Optional[Callable] = None,
        batched: bool = True,
        num_proc: int = 4,
    ) -> Any:
        """应用预处理到数据集"""
        if tokenize_func is None:
            tokenize_func = self.tokenize_function

        print("🔄 开始数据预处理...")
        tokenized_dataset = dataset.map(
            tokenize_func,
            batched=batched,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
        )
        print(f"✅ 预处理完成，数据集大小: {len(tokenized_dataset)}")

        return tokenized_dataset


class DatasetSplitter:
    """数据集分割器"""

    @staticmethod
    def split_train_test(
        dataset: Any,
        test_size: float = 0.1,
        seed: int = 42,
    ) -> Tuple[Any, Any]:
        """分割为训练集和测试集"""
        print(f"✂️  分割数据集，测试集比例: {test_size}")
        split = dataset.train_test_split(test_size=test_size, seed=seed)
        print(f"   训练集: {len(split['train'])} 条")
        print(f"   测试集: {len(split['test'])} 条")
        return split["train"], split["test"]


# ============================================
# 示例数据集模板
# ============================================

def create_sample_instruction_dataset(
    output_path: str = "datasets/sample_instructions.jsonl",
    num_samples: int = 100,
) -> str:
    """创建示例指令微调数据集"""
    import random

    samples = []

    # 示例指令模板
    instructions = [
        "解释以下概念",
        "总结这段文本",
        "翻译为英文",
        "写一段关于{}的介绍",
        "列出{}的步骤",
        "比较{}和{}的区别",
        "用简单的语言解释{}",
        "给出{}的例子",
        "回答以下问题",
        "为以下内容写一个标题",
    ]

    topics = [
        "机器学习", "深度学习", "神经网络", "自然语言处理",
        "计算机视觉", "数据科学", "云计算", "人工智能",
        "大语言模型", "Transformer架构",
    ]

    for i in range(num_samples):
        topic = random.choice(topics)
        instruction = random.choice(instructions).format(topic, random.choice(topics))

        samples.append({
            "instruction": instruction,
            "input": "",
            "output": f"这是关于{topic}的示例回答。在实际使用中，这里应该包含高质量的人工标注。",
        })

    # 保存为 JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"✅ 示例数据集已创建: {output_path}")
    print(f"   包含 {num_samples} 条样本（用于演示，实际使用需要真实数据）")
    return output_path
