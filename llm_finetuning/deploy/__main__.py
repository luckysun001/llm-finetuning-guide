"""``python -m llm_finetuning.deploy`` 命令行入口。

把 :func:`llm_finetuning.deploy.server.run_server` 包装成可直接使用的命令，
省去"为了起个服务还得写一段 Python"的麻烦。

用法::

    python -m llm_finetuning.deploy --model-path ./checkpoints/qlora-7b
    python -m llm_finetuning.deploy --model-path ./exports/gguf --backend vllm --port 8000

注意：真正的启动依赖 ``fastapi`` 与 ``uvicorn``（未安装时这里会给出安装提示，
不会静默失败），加载模型还需要 ``transformers`` 或 ``vllm``。
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llm_finetuning.deploy",
        description="启动 LLM 微调产物的推理服务（FastAPI）",
    )
    parser.add_argument("--model-path", required=True, help="模型权重目录")
    parser.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "vllm"],
        help="推理后端（默认 transformers）",
    )
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"], help="推理设备"
    )
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"], help="权重精度"
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="默认最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="默认采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="默认 top-p")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM 后端的显存占用比例（默认 0.9）",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    from llm_finetuning.deploy import DeployConfig, run_server

    config = DeployConfig(
        model_path=args.model_path,
        backend=args.backend,
        host=args.host,
        port=args.port,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    try:
        run_server(config)
    except ImportError as exc:
        print(f"缺少依赖: {exc}", file=sys.stderr)
        print(
            "请安装服务依赖: pip install fastapi uvicorn"
            + (" vllm" if args.backend == "vllm" else " torch transformers"),
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\n已停止服务")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
