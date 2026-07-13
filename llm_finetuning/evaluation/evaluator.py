"""
模型评估模块
"""

import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    """评估结果"""
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_scores: Dict[str, float] = field(default_factory=dict)
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    sample_predictions: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perplexity": self.perplexity,
            "bleu_score": self.bleu_score,
            "rouge_scores": self.rouge_scores,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            **self.custom_metrics,
        }

    def __repr__(self) -> str:
        metrics = []
        if self.perplexity:
            metrics.append(f"PPL={self.perplexity:.2f}")
        if self.bleu_score:
            metrics.append(f"BLEU={self.bleu_score:.2f}")
        if self.accuracy:
            metrics.append(f"Acc={self.accuracy:.2%}")
        return f"EvaluationResult({', '.join(metrics)})"


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def compute_perplexity(
        self,
        eval_dataset,
        batch_size: int = 8,
        max_samples: Optional[int] = None,
    ) -> float:
        """计算困惑度 (Perplexity)"""
        import math
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        print("📊 计算困惑度...")

        if max_samples:
            eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

        # 创建数据加载器
        dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        total_loss = 0
        num_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估中"):
                # 准备输入
                inputs = {
                    "input_ids": torch.stack(batch["input_ids"], dim=1).to(self.model.device),
                    "attention_mask": torch.stack(batch["attention_mask"], dim=1).to(self.model.device),
                    "labels": torch.stack(batch["labels"], dim=1).to(self.model.device),
                }

                # 前向传播
                outputs = self.model(**inputs)
                loss = outputs.loss

                # 累积损失
                num_tokens += inputs["attention_mask"].sum().item()
                total_loss += loss.item() * inputs["attention_mask"].sum().item()

        perplexity = math.exp(total_loss / num_tokens) if num_tokens > 0 else float('inf')
        print(f"✅ 困惑度: {perplexity:.2f}")
        return perplexity

    def compute_bleu(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> float:
        """计算 BLEU 分数"""
        try:
            from evaluate import load

            print("📊 计算 BLEU 分数...")
            bleu = load("bleu")
            results = bleu.compute(predictions=predictions, references=references)
            bleu_score = results["bleu"]
            print(f"✅ BLEU 分数: {bleu_score:.4f}")
            return bleu_score

        except ImportError:
            print("⚠️  evaluate 模块未安装，跳过 BLEU 计算")
            return None

    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """计算 ROUGE 分数"""
        try:
            from evaluate import load

            print("📊 计算 ROUGE 分数...")
            rouge = load("rouge")
            results = rouge.compute(predictions=predictions, references=references)
            print(f"✅ ROUGE-1: {results['rouge1']:.4f}, ROUGE-2: {results['rouge2']:.4f}, "
                  f"ROUGE-L: {results['rougeL']:.4f}")
            return results

        except ImportError:
            print("⚠️  evaluate 模块未安装，跳过 ROUGE 计算")
            return {}

    def generate_samples(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        num_samples: int = 5,
    ) -> List[Dict[str, str]]:
        """生成示例预测"""
        import torch

        print(f"🔮 生成 {num_samples} 个示例预测...")
        samples = []

        for i, prompt in enumerate(prompts[:num_samples]):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            samples.append({
                "prompt": prompt,
                "prediction": generated_text[len(prompt):],
            })

        print(f"✅ 示例生成完成")
        return samples

    def evaluate(
        self,
        eval_dataset,
        compute_ppl: bool = True,
        generate_samples: bool = True,
        num_samples: int = 5,
    ) -> EvaluationResult:
        """完整评估"""
        result = EvaluationResult()

        # 计算困惑度
        if compute_ppl:
            result.perplexity = self.compute_perplexity(eval_dataset)

        # 生成示例（用于人工评估）
        if generate_samples and hasattr(eval_dataset, '__getitem__'):
            # 提取几个示例
            prompts = []
            for i in range(min(num_samples, len(eval_dataset))):
                # 解码输入ID为文本
                input_ids = eval_dataset[i]["input_ids"]
                prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                prompts.append(prompt[:100] + "..." if len(prompt) > 100 else prompt)

            result.sample_predictions = self.generate_samples(prompts, num_samples=num_samples)

        return result

    def save_results(self, result: EvaluationResult, output_path: str) -> None:
        """保存评估结果"""
        import json

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"✅ 评估结果已保存到: {output_path}")


class ComparisonEvaluator:
    """模型对比评估器"""

    def __init__(self, models: Dict[str, Any], tokenizer: Any):
        self.models = models
        self.tokenizer = tokenizer

    def compare_perplexity(
        self,
        eval_dataset,
    ) -> Dict[str, float]:
        """对比多个模型的困惑度"""
        results = {}

        for name, model in self.models.items():
            print(f"\n📊 评估模型: {name}")
            evaluator = ModelEvaluator(model, self.tokenizer)
            ppl = evaluator.compute_perplexity(eval_dataset)
            results[name] = ppl

        # 打印对比结果
        print("\n" + "=" * 50)
        print("📊 困惑度对比 (越低越好)")
        print("=" * 50)
        for name, ppl in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name}: {ppl:.2f}")
        print("=" * 50)

        return results
