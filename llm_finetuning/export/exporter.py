"""
模型导出模块
============

把微调后的模型导出为可直接部署的格式：

- ``safetensors``：HuggingFace 标准权重（含 tokenizer），后续可被 vLLM / TGI 直接加载
- ``onnx``：ONNX 计算图，便于在 ONNX Runtime / TensorRT 上推理
- ``gguf``：llama.cpp / Ollama 使用的量化格式

设计约定：torch、transformers、onnx 等重型依赖**只在函数内部按需导入**，
这样只做配置与规划（不真正导出）时无需安装它们。
"""

import hashlib
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_finetuning.config import QuantizationLevel


class ExportFormat(str, Enum):
    """支持的导出格式"""

    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    GGUF = "gguf"


#: 每个参数占用的字节数（用于体积估算）
DTYPE_BYTES: Dict[str, float] = {
    QuantizationLevel.FP32.value: 4.0,
    QuantizationLevel.FP16.value: 2.0,
    QuantizationLevel.BF16.value: 2.0,
    QuantizationLevel.INT8.value: 1.0,
    QuantizationLevel.INT4.value: 0.5,
}


@dataclass
class ExportConfig:
    """导出配置"""

    output_dir: str = "./exports"
    export_format: str = ExportFormat.SAFETENSORS.value
    #: 导出时应用的量化级别，None 表示保持训练精度
    quantization: Optional[str] = None
    #: 是否一并保存 tokenizer
    save_tokenizer: bool = True
    #: ONNX opset 版本
    opset: int = 14
    #: GGUF 量化方案（q4_k_m / q5_k_m / q8_0 / f16 等）
    gguf_quant: str = "q4_k_m"
    #: 是否把分片权重合并为单文件
    merge_shards: bool = True

    def validate(self) -> None:
        """校验配置，非法值直接抛错（避免导出到一半才失败）"""
        try:
            ExportFormat(self.export_format)
        except ValueError as exc:
            raise ValueError(
                f"不支持的导出格式: {self.export_format}，"
                f"可选: {[f.value for f in ExportFormat]}"
            ) from exc

        if self.quantization is not None:
            try:
                QuantizationLevel(self.quantization)
            except ValueError as exc:
                raise ValueError(
                    f"不支持的量化级别: {self.quantization}，"
                    f"可选: {[q.value for q in QuantizationLevel]}"
                ) from exc

        if self.opset < 7:
            raise ValueError(f"ONNX opset 过小: {self.opset}（至少 7）")


@dataclass
class ExportResult:
    """导出结果"""

    export_format: str
    output_dir: str
    files: List[str] = field(default_factory=list)
    quantization: Optional[str] = None
    size_mb: float = 0.0
    #: 需要用户手动执行的后续命令（例如 gguf 转换）
    follow_up_command: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "export_format": self.export_format,
            "output_dir": self.output_dir,
            "files": self.files,
            "quantization": self.quantization,
            "size_mb": round(self.size_mb, 2),
            "follow_up_command": self.follow_up_command,
        }


def _require(module: str, extra: str):
    """按需导入重型依赖，缺失时给出可执行的安装提示"""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"导出需要 `{module}`，请先安装：pip install {extra}"
        ) from exc


class ModelExporter:
    """模型导出器"""

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        self.config.validate()

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def export(self, model: Any, tokenizer: Any = None) -> ExportResult:
        """按配置导出模型

        Args:
            model: 微调后的模型对象（HF PreTrainedModel 或 PeftModel）
            tokenizer: 对应的 tokenizer，可选

        Returns:
            ExportResult
        """
        fmt = ExportFormat(self.config.export_format)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if fmt == ExportFormat.SAFETENSORS:
            result = self._export_safetensors(model, tokenizer, output_dir)
        elif fmt == ExportFormat.ONNX:
            result = self._export_onnx(model, tokenizer, output_dir)
        else:
            result = self._export_gguf(model, tokenizer, output_dir)

        result.size_mb = self._dir_size_mb(output_dir)
        return result

    # ------------------------------------------------------------------
    # 各格式实现
    # ------------------------------------------------------------------
    def _export_safetensors(
        self, model: Any, tokenizer: Any, output_dir: Path
    ) -> ExportResult:
        """导出 HF 标准权重目录（safetensors + config + tokenizer）"""
        self._apply_quantization(model)

        model.save_pretrained(
            str(output_dir), safe_serialization=True, max_shard_size="5GB"
        )
        files = self._list_files(output_dir)

        if tokenizer is not None and self.config.save_tokenizer:
            tokenizer.save_pretrained(str(output_dir))
            files = self._list_files(output_dir)

        return ExportResult(
            export_format=ExportFormat.SAFETENSORS.value,
            output_dir=str(output_dir),
            files=files,
            quantization=self.config.quantization,
        )

    def _export_onnx(self, model: Any, tokenizer: Any, output_dir: Path) -> ExportResult:
        """导出 ONNX 计算图"""
        torch = _require("torch", "torch")
        _require("onnx", "onnx")

        onnx_path = output_dir / "model.onnx"
        model.eval()

        # 构造 dummy 输入：优先从模型配置推断，失败则用通用 1x8 输入
        batch, seq = 1, 8
        try:
            seq = int(getattr(model.config, "max_position_embeddings", seq))
            seq = min(seq, 512)  # 避免 dummy 输入过大导致导出极慢
        except Exception:
            pass

        dummy = torch.zeros((batch, seq), dtype=torch.long)
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=self.config.opset,
        )

        if tokenizer is not None and self.config.save_tokenizer:
            tokenizer.save_pretrained(str(output_dir))

        return ExportResult(
            export_format=ExportFormat.ONNX.value,
            output_dir=str(output_dir),
            files=self._list_files(output_dir),
            quantization=self.config.quantization,
        )

    def _export_gguf(self, model: Any, tokenizer: Any, output_dir: Path) -> ExportResult:
        """准备 GGUF 导出

        GGUF 转换依赖 llama.cpp 的 ``convert_hf_to_gguf.py`` 脚本（Python 侧无法直接生成），
        因此这里先把 HF 权重落盘，再返回一条可直接执行的转换命令给用户。
        """
        # 先导出标准 HF 权重，作为 llama.cpp 的输入
        hf_dir = output_dir / "hf"
        hf_result = self._export_safetensors(model, tokenizer, hf_dir)
        hf_result.export_format = ExportFormat.GGUF.value

        gguf_path = output_dir / f"model-{self.config.gguf_quant}.gguf"
        hf_result.follow_up_command = (
            f"python llama.cpp/convert_hf_to_gguf.py {hf_dir} "
            f"--outfile {gguf_path} --outtype {self.config.gguf_quant}"
        )
        return hf_result

    # ------------------------------------------------------------------
    # 量化与工具
    # ------------------------------------------------------------------
    def _apply_quantization(self, model: Any) -> None:
        """按配置对模型做量化（int4/int8 需要 bitsandbytes）"""
        level = self.config.quantization
        if level in (None, QuantizationLevel.FP32.value):
            return

        if level == QuantizationLevel.INT8.value:
            _require("bitsandbytes", "bitsandbytes")
            if hasattr(model, "quantize"):
                model.quantize(8)
            return

        if level == QuantizationLevel.INT4.value:
            _require("bitsandbytes", "bitsandbytes")
            if hasattr(model, "quantize"):
                model.quantize(4)
            return

        # fp16 / bf16：转换参数 dtype
        torch = _require("torch", "torch")
        dtype_map = {
            QuantizationLevel.FP16.value: torch.float16,
            QuantizationLevel.BF16.value: torch.bfloat16,
        }
        if level in dtype_map and hasattr(model, "to"):
            model.to(dtype_map[level])

    @staticmethod
    def estimate_size_mb(num_params: int, quantization: str = "fp16") -> float:
        """估算模型体积（MB）

        Args:
            num_params: 参数量，例如 7e9
            quantization: fp32 / fp16 / bf16 / int8 / int4

        Returns:
            估算体积（MB）
        """
        try:
            bytes_per_param = DTYPE_BYTES[QuantizationLevel(quantization).value]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"不支持的量化级别: {quantization}") from exc

        return num_params * bytes_per_param / (1024 ** 2)

    @staticmethod
    def _list_files(directory: Path) -> List[str]:
        return sorted(
            str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
        )

    @staticmethod
    def _dir_size_mb(directory: Path) -> float:
        total = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        return total / (1024 ** 2)


def fingerprint_directory(directory: str) -> str:
    """对导出目录做指纹校验（用于确认导出产物一致性）"""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def copy_to_dist(src: str, dst: str) -> str:
    """把导出结果复制到发布目录"""
    source, target = Path(src), Path(dst)
    if not source.exists():
        raise FileNotFoundError(f"源目录不存在: {src}")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    return str(target)
