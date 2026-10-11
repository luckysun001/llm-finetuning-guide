"""模型导出模块测试

不依赖 torch / transformers：只验证配置校验、体积估算与命令生成等纯逻辑，
以及「缺依赖时给出可执行安装提示」的错误路径。
"""
import pytest

from llm_finetuning.export.exporter import (
    ExportConfig,
    ExportFormat,
    ExportResult,
    ModelExporter,
    copy_to_dist,
    fingerprint_directory,
)


class TestExportConfig:
    """配置校验"""

    def test_defaults(self):
        cfg = ExportConfig()
        cfg.validate()
        assert cfg.export_format == "safetensors"
        assert cfg.save_tokenizer is True

    @pytest.mark.parametrize("fmt", ["safetensors", "onnx", "gguf"])
    def test_supported_formats(self, fmt):
        ExportConfig(export_format=fmt).validate()

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="不支持的导出格式"):
            ExportConfig(export_format="tensorrt").validate()

    def test_unknown_quantization_raises(self):
        with pytest.raises(ValueError, match="不支持的量化级别"):
            ExportConfig(quantization="int2").validate()

    def test_small_opset_raises(self):
        with pytest.raises(ValueError, match="opset"):
            ExportConfig(export_format="onnx", opset=3).validate()

    @pytest.mark.parametrize("level", ["fp32", "fp16", "bf16", "int8", "int4"])
    def test_supported_quantization(self, level):
        ExportConfig(quantization=level).validate()


class TestEstimateSize:
    """体积估算"""

    def test_fp16_7b(self):
        # 7e9 参数 × 2 字节 ≈ 13363 MB
        assert ModelExporter.estimate_size_mb(7e9, "fp16") == pytest.approx(
            7e9 * 2 / 1024 ** 2, rel=1e-6
        )

    def test_quantization_reduces_size_monotonically(self):
        sizes = [
            ModelExporter.estimate_size_mb(7e9, level)
            for level in ("fp32", "fp16", "int8", "int4")
        ]
        assert sizes == sorted(sizes, reverse=True)

    def test_int4_is_quarter_of_fp16(self):
        assert ModelExporter.estimate_size_mb(1e9, "int4") == pytest.approx(
            ModelExporter.estimate_size_mb(1e9, "fp16") / 4, rel=1e-6
        )

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError):
            ModelExporter.estimate_size_mb(1e9, "int2")


class TestGGUFCommand:
    """GGUF 导出：落盘 HF 权重 + 给出 llama.cpp 转换命令"""

    def test_gguf_export_returns_follow_up_command(self, tmp_path):
        exporter = ModelExporter(
            ExportConfig(
                output_dir=str(tmp_path / "out"),
                export_format=ExportFormat.GGUF.value,
                gguf_quant="q5_k_m",
            )
        )

        result = exporter.export(_FakeModel(), _FakeTokenizer())

        assert result.export_format == "gguf"
        assert result.follow_up_command is not None
        assert "convert_hf_to_gguf.py" in result.follow_up_command
        assert "q5_k_m" in result.follow_up_command
        # HF 权重已作为转换输入落盘
        assert (tmp_path / "out" / "hf" / "model.safetensors").exists()

    def test_safetensors_export_saves_tokenizer(self, tmp_path):
        exporter = ModelExporter(
            ExportConfig(output_dir=str(tmp_path / "sft"), save_tokenizer=True)
        )

        result = exporter.export(_FakeModel(), _FakeTokenizer())

        assert "model.safetensors" in result.files
        assert "tokenizer_config.json" in result.files
        assert result.follow_up_command is None

    def test_export_without_tokenizer(self, tmp_path):
        exporter = ModelExporter(
            ExportConfig(output_dir=str(tmp_path / "sft2"), save_tokenizer=False)
        )
        result = exporter.export(_FakeModel())
        assert "tokenizer_config.json" not in result.files


class TestDependencyHints:
    """缺依赖时给出可执行的安装提示"""

    def test_onnx_export_requires_torch(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        exporter = ModelExporter(
            ExportConfig(output_dir=str(tmp_path / "onnx"), export_format="onnx")
        )
        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="pip install torch"):
            exporter.export(_FakeModel(), _FakeTokenizer())


class TestUtils:
    """目录指纹与发布复制"""

    def test_fingerprint_is_stable_and_content_sensitive(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
        first = fingerprint_directory(str(tmp_path))
        assert first == fingerprint_directory(str(tmp_path))

        (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
        assert fingerprint_directory(str(tmp_path)) != first

    def test_fingerprint_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            fingerprint_directory("./definitely-not-exists")

    def test_copy_to_dist(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "model.safetensors").write_text("w", encoding="utf-8")

        dst = copy_to_dist(str(src), str(tmp_path / "dist"))
        assert (tmp_path / "dist" / "model.safetensors").exists()
        assert dst == str(tmp_path / "dist")


class TestExportResult:
    def test_to_dict_rounds_size(self):
        result = ExportResult(
            export_format="safetensors", output_dir="/tmp", size_mb=1.23456
        )
        assert result.to_dict()["size_mb"] == 1.23


# ----------------------------------------------------------------------
# 测试替身：模拟 HF 模型的 save_pretrained 接口
# ----------------------------------------------------------------------
class _FakeModel:
    def __init__(self):
        self.saved = False

    def save_pretrained(self, path, **kwargs):
        from pathlib import Path

        self.saved = True
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.safetensors").write_bytes(b"weights")
        (Path(path) / "config.json").write_text("{}", encoding="utf-8")

    def to(self, *args, **kwargs):
        return self


class _FakeTokenizer:
    def save_pretrained(self, path, **kwargs):
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")
