"""
LoRA 微调方法实现
Low-Rank Adaptation of Large Language Models
"""

import os
from typing import Any, Dict, Optional, Union

from llm_finetuning.config import Config, LoRAConfig, QLoRAConfig


class LoRAFinetuner:
    """LoRA 微调器"""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_config = None

    def setup_peft_config(self) -> Any:
        """设置 PEFT 配置"""
        try:
            from peft import LoraConfig, TaskType

            # 根据配置创建 LoRA 配置
            lora_config = getattr(self.config, 'lora_config', LoRAConfig())

            self.peft_config = LoraConfig(
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                task_type=TaskType.CAUSAL_LM,
            )

            return self.peft_config

        except ImportError:
            raise ImportError(
                "peft 模块未安装，请执行: pip install peft"
            )

    def load_model_and_tokenizer(self) -> tuple:
        """加载预训练模型和分词器"""
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
            import torch

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 量化配置（如果是 QLoRA）
            quantization_config = None
            if hasattr(self.config, 'lora_config') and isinstance(self.config.lora_config, QLoRAConfig):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=self.config.lora_config.load_in_4bit,
                    bnb_4bit_use_double_quant=self.config.lora_config.bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=self.config.lora_config.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.lora_config.bnb_4bit_compute_dtype),
                )

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.config.training.bf16 else torch.float16,
            )

            return self.model, self.tokenizer

        except ImportError as e:
            raise ImportError(
                f"transformers 或 torch 未安装: {e}\n"
                f"请执行: pip install torch transformers"
            )

    def apply_lora(self) -> Any:
        """应用 LoRA 适配器到模型"""
        if self.peft_config is None:
            self.setup_peft_config()

        try:
            from peft import get_peft_model

            self.model = get_peft_model(self.model, self.peft_config)
            print("✅ LoRA 适配器已应用")
            self.model.print_trainable_parameters()

            return self.model

        except ImportError:
            raise ImportError(
                "peft 模块未安装，请执行: pip install peft"
            )

    def train(self, train_dataset, eval_dataset=None, callbacks=None):
        """训练模型"""
        try:
            from transformers import TrainingArguments, Trainer
            from trl import SFTTrainer

            # 设置训练参数
            training_args = TrainingArguments(
                output_dir=self.config.training.output_dir,
                num_train_epochs=self.config.training.num_train_epochs,
                per_device_train_batch_size=self.config.training.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                warmup_steps=self.config.training.warmup_steps,
                fp16=self.config.training.fp16,
                bf16=self.config.training.bf16,
                logging_steps=self.config.training.logging_steps,
                save_strategy=self.config.training.save_strategy,
                evaluation_strategy=self.config.training.evaluation_strategy if eval_dataset else "no",
                save_total_limit=self.config.training.save_total_limit,
                load_best_model_at_end=eval_dataset is not None,
                report_to=self.config.training.report_to,
                gradient_checkpointing=self.config.training.gradient_checkpointing,
            )

            # 使用 SFTTrainer 进行监督微调
            trainer = SFTTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=self.peft_config,
                dataset_text_field=self.config.data.text_column,
                max_seq_length=self.config.data.max_seq_length,
                tokenizer=self.tokenizer,
                args=training_args,
                callbacks=callbacks,
            )

            # 开始训练
            print(f"🚀 开始训练，学习率: {self.config.training.learning_rate}")
            trainer.train()

            return trainer

        except ImportError as e:
            raise ImportError(
                f"trl 或 transformers 模块未安装: {e}\n"
                f"请执行: pip install trl transformers accelerate"
            )

    def save_model(self, output_dir: Optional[str] = None):
        """保存 LoRA 适配器"""
        save_dir = output_dir or self.config.training.output_dir
        os.makedirs(save_dir, exist_ok=True)

        # 只保存 LoRA 适配器（文件很小，通常几 MB）
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        print(f"✅ 模型已保存到: {save_dir}")
        print(f"   注意: 只保存了 LoRA 适配器，不是完整模型")

    def merge_and_save(self, output_dir: str):
        """合并 LoRA 适配器并保存完整模型"""
        print("🔄 合并 LoRA 适配器到基础模型...")

        # 转换为完整模型
        merged_model = self.model.merge_and_unload()

        # 保存完整模型
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"✅ 合并后的完整模型已保存到: {output_dir}")


class QLoRAFinetuner(LoRAFinetuner):
    """
    QLoRA 微调器（4位量化 LoRA）
    在保持性能的同时大幅降低显存需求
    """

    def __init__(self, config: Config):
        super().__init__(config)
        # 确保使用 QLoRA 配置
        if not hasattr(config, 'lora_config') or not isinstance(config.lora_config, QLoRAConfig):
            config.lora_config = QLoRAConfig()
            config.quantization_level = "int4"

    def setup_peft_config(self) -> Any:
        """设置 QLoRA 配置"""
        from peft import LoraConfig, TaskType

        lora_config = self.config.lora_config
        self.peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            target_modules=lora_config.target_modules,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        return self.peft_config
