"""
模型部署模块
============

把微调后的模型以 HTTP 服务的方式跑起来：

- ``transformers`` 后端：依赖最少，适合本地验证
- ``vllm`` 后端：吞吐高，适合生产部署

设计约定：fastapi / uvicorn / torch / vllm 均**在函数内部按需导入**，
未安装时给出明确的安装提示，而不是在 import 本模块时就炸掉。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DeployConfig:
    """部署配置"""

    model_path: str
    backend: str = "transformers"  # transformers | vllm
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "auto"           # auto | cuda | cpu
    dtype: str = "auto"            # auto | fp16 | bf16 | fp32
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    #: vLLM 专用：显存利用率
    gpu_memory_utilization: float = 0.9
    #: 服务路径
    api_prefix: str = ""

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("model_path 不能为空")
        if self.backend not in ("transformers", "vllm"):
            raise ValueError(
                f"不支持的推理后端: {self.backend}，可选: transformers / vllm"
            )
        if not (0 < self.temperature <= 2.0):
            raise ValueError(f"temperature 应在 (0, 2] 之间: {self.temperature}")
        if not (0 < self.top_p <= 1.0):
            raise ValueError(f"top_p 应在 (0, 1] 之间: {self.top_p}")
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"非法端口: {self.port}")


@dataclass
class GenerationRequest:
    """生成请求"""

    prompt: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationRequest":
        if "prompt" not in data:
            raise ValueError("请求体缺少 prompt 字段")
        return cls(
            prompt=str(data["prompt"]),
            max_new_tokens=data.get("max_new_tokens"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
        )


@dataclass
class GenerationResponse:
    """生成响应"""

    text: str
    model: str
    backend: str
    finish_reason: str = "length"
    usage: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "backend": self.backend,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }


class InferenceBackend(ABC):
    """推理后端抽象"""

    #: 后端名称
    name: str = "base"

    @abstractmethod
    def load(self, config: DeployConfig) -> None:
        """加载模型"""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """执行一次生成"""


class TransformersBackend(InferenceBackend):
    """基于 transformers 的推理后端"""

    name = "transformers"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._config: Optional[DeployConfig] = None

    def load(self, config: DeployConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "auto": "auto",
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.dtype, "auto")

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch_dtype,
            device_map=config.device,
            trust_remote_code=True,
        )
        self._config = config

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        # 先做状态检查，避免未加载模型时因缺少 torch 而抛出误导性的 ImportError
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        import torch

        cfg = self._config or DeployConfig(model_path="")
        inputs = self._tokenizer(request.prompt, return_tensors="pt")
        if cfg.device != "cpu":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        max_new_tokens = request.max_new_tokens or cfg.max_new_tokens
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature or cfg.temperature,
                top_p=request.top_p or cfg.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][prompt_len:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)

        return GenerationResponse(
            text=text,
            model=cfg.model_path,
            backend=self.name,
            finish_reason="length" if len(generated) >= max_new_tokens else "stop",
            usage={
                "prompt_tokens": int(prompt_len),
                "completion_tokens": int(len(generated)),
                "total_tokens": int(prompt_len + len(generated)),
            },
        )


class VLLMBackend(InferenceBackend):
    """基于 vLLM 的高吞吐推理后端"""

    name = "vllm"

    def __init__(self):
        self._llm = None
        self._config: Optional[DeployConfig] = None

    def load(self, config: DeployConfig) -> None:
        from vllm import LLM

        self._llm = LLM(
            model=config.model_path,
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )
        self._config = config

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._llm is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        from vllm import SamplingParams

        cfg = self._config or DeployConfig(model_path="")
        params = SamplingParams(
            max_tokens=request.max_new_tokens or cfg.max_new_tokens,
            temperature=request.temperature or cfg.temperature,
            top_p=request.top_p or cfg.top_p,
        )
        outputs = self._llm.generate([request.prompt], params)
        output = outputs[0].outputs[0]

        return GenerationResponse(
            text=output.text,
            model=cfg.model_path,
            backend=self.name,
            finish_reason=output.finish_reason or "stop",
            usage={
                "prompt_tokens": len(outputs[0].prompt_token_ids),
                "completion_tokens": len(output.token_ids),
                "total_tokens": len(outputs[0].prompt_token_ids) + len(output.token_ids),
            },
        )


def build_backend(backend: str) -> InferenceBackend:
    """按名称创建推理后端

    Args:
        backend: transformers | vllm
    """
    if backend == "transformers":
        return TransformersBackend()
    if backend == "vllm":
        return VLLMBackend()
    raise ValueError(f"不支持的推理后端: {backend}，可选: transformers / vllm")


def create_app(config: DeployConfig, backend: Optional[InferenceBackend] = None):
    """创建 FastAPI 应用

    提供三个端点：
    - ``GET /health``：健康检查
    - ``GET /v1/info``：模型与后端信息
    - ``POST /v1/generate``：文本生成
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as exc:
        raise ImportError(
            "部署服务需要 fastapi，请先安装：pip install fastapi uvicorn"
        ) from exc

    config.validate()
    engine = backend or build_backend(config.backend)
    engine.load(config)

    app = FastAPI(
        title="LLM Finetuning Serving",
        version="1.0.0",
        description="微调模型推理服务",
    )

    class GeneratePayload(BaseModel):
        prompt: str
        max_new_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "backend": engine.name}

    @app.get("/v1/info")
    async def info() -> Dict[str, Any]:
        return {
            "model": config.model_path,
            "backend": engine.name,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

    @app.post("/v1/generate")
    async def generate(payload: GeneratePayload) -> Dict[str, Any]:
        try:
            request = GenerationRequest(
                prompt=payload.prompt,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            return engine.generate(request).to_dict()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def run_server(config: DeployConfig, backend: Optional[InferenceBackend] = None) -> None:
    """启动推理服务（阻塞）"""
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("启动服务需要 uvicorn，请先安装：pip install uvicorn") from exc

    app = create_app(config, backend=backend)
    uvicorn.run(app, host=config.host, port=config.port)


def list_backends() -> List[str]:
    """列出可用的推理后端名称"""
    return ["transformers", "vllm"]
