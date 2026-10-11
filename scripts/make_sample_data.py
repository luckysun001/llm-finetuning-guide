"""
生成示例指令微调数据集
======================

仓库的 .gitignore 忽略了 datasets/*.jsonl（避免大文件入库），
因此示例数据通过本脚本生成到本地。

用法:
    python scripts/make_sample_data.py
    python scripts/make_sample_data.py --output datasets/my_data.jsonl
"""

import argparse
import json
from pathlib import Path

#: 10 条中文指令样本，字段与 DataPreprocessor 的指令模板对应
SAMPLES = [
    {
        "instruction": "用一句话解释什么是 LoRA 微调",
        "input": "",
        "output": "LoRA 通过在原始权重旁插入低秩矩阵来训练，只更新极少量参数即可适配下游任务，显存和算力开销远低于全参微调。",
    },
    {
        "instruction": "把下面这句话改写成更正式的表达",
        "input": "这个模型跑得有点慢，能不能想办法弄快点",
        "output": "当前模型推理延迟较高，建议通过量化、算子融合或批处理等方式优化吞吐。",
    },
    {
        "instruction": "解释过拟合的常见原因与缓解手段",
        "input": "",
        "output": "过拟合通常源于数据量不足、模型容量过大或训练轮数过多；可通过增加数据、数据增强、权重衰减、早停和 dropout 缓解。",
    },
    {
        "instruction": "为下面的函数写一段简短的文档字符串",
        "input": "def add(a, b): return a + b",
        "output": "返回两个数的和。参数 a、b 为数值类型，返回值为数值类型。",
    },
    {
        "instruction": "列出选择微调学习率的三个经验原则",
        "input": "",
        "output": "一、全参微调用小学习率（1e-5 量级），LoRA/QLoRA 可用更大学习率（1e-4 量级）；二、batch size 增大时可适当提高学习率；三、配合 warmup 与余弦衰减更稳定。",
    },
    {
        "instruction": "把下面的技术术语翻译成中文并简要说明",
        "input": "quantization-aware training",
        "output": "量化感知训练：在训练过程中模拟量化带来的误差，使模型在真正量化部署后精度损失更小。",
    },
    {
        "instruction": "说明 4-bit 量化相比 8-bit 的取舍",
        "input": "",
        "output": "4-bit 显存占用更低、可跑更大模型，但精度损失更大、对校准数据更敏感；8-bit 精度更好、兼容性更强，显存节省约为一半。",
    },
    {
        "instruction": "概括一份模型评估报告应包含哪些内容",
        "input": "",
        "output": "应包含评估数据集与划分方式、客观指标（困惑度、准确率等）、与基线模型的对比、人工抽样结果、典型失败案例以及结论与改进建议。",
    },
    {
        "instruction": "把下面的需求拆成三个可执行的子任务",
        "input": "搭建一个企业内部知识问答助手",
        "output": "一、梳理知识源并完成文档清洗与切分；二、搭建检索链路（向量库 + 召回 + 重排）并评测召回率；三、接入大模型生成答案，设计引用与拒答策略。",
    },
    {
        "instruction": "解释为什么微调后要保留一份测试集不参与训练",
        "input": "",
        "output": "保留独立测试集可以无偏地估计模型的泛化能力，避免用训练数据自评导致指标虚高，从而及时发现过拟合与数据泄漏。",
    },
]


def write_jsonl(samples, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成示例指令微调数据集")
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/sample_instructions.jsonl",
        help="输出文件路径（默认 datasets/sample_instructions.jsonl）",
    )
    args = parser.parse_args()

    count = write_jsonl(SAMPLES, Path(args.output))
    print(f"✅ 已生成 {count} 条示例数据 -> {args.output}")


if __name__ == "__main__":
    main()
