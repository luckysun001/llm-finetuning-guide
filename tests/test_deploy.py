"""模型部署模块测试

覆盖配置校验、后端选择、请求/响应模型与错误路径。
真实推理需要 torch / vllm，这里用测试替身验证调用契约。
"""
import pytest

from llm_finetuning.deploy.server import (
    DeployConfig,
    GenerationRequest,
    GenerationResponse,
    InferenceBackend,
    TransformersBackend,
    VLLMBackend,
    build_backend,
    create_app,
    list_backends,
    run_server,
)


class TestDeployConfig:
    """配置校验"""

    def test_defaults(self):
        cfg = DeployConfig(model_path="/models/qwen")
        cfg.validate()
        assert cfg.backend == "transformers"
        assert cfg.port == 8000

    def test_empty_model_path_raises(self):
        with pytest.raises(ValueError, match="model_path"):
            DeployConfig(model_path="").validate()

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="推理后端"):
            DeployConfig(model_path="m", backend="ggml").validate()

    @pytest.mark.parametrize("temp", [0.0, 2.5])
    def test_temperature_range(self, temp):
        with pytest.raises(ValueError, match="temperature"):
            DeployConfig(model_path="m", temperature=temp).validate()

    @pytest.mark.parametrize("top_p", [0.0, 1.5])
    def test_top_p_range(self, top_p):
        with pytest.raises(ValueError, match="top_p"):
            DeployConfig(model_path="m", top_p=top_p).validate()

    @pytest.mark.parametrize("port", [0, 70000])
    def test_port_range(self, port):
        with pytest.raises(ValueError, match="端口"):
            DeployConfig(model_path="m", port=port).validate()


class TestBackends:
    """后端构造与契约"""

    def test_build_transformers(self):
        backend = build_backend("transformers")
        assert isinstance(backend, TransformersBackend)
        assert backend.name == "transformers"

    def test_build_vllm(self):
        backend = build_backend("vllm")
        assert isinstance(backend, VLLMBackend)
        assert backend.name == "vllm"

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError):
            build_backend("nope")

    def test_list_backends(self):
        assert set(list_backends()) == {"transformers", "vllm"}

    def test_backends_are_inference_backends(self):
        for backend in (TransformersBackend(), VLLMBackend()):
            assert isinstance(backend, InferenceBackend)

    def test_generate_before_load_raises(self):
        backend = TransformersBackend()
        with pytest.raises(RuntimeError, match="模型未加载"):
            backend.generate(GenerationRequest(prompt="hi"))


class TestRequestResponse:
    """请求与响应模型"""

    def test_from_dict(self):
        req = GenerationRequest.from_dict(
            {"prompt": "你好", "max_new_tokens": 32, "temperature": 0.1}
        )
        assert req.prompt == "你好"
        assert req.max_new_tokens == 32
        assert req.temperature == 0.1

    def test_from_dict_missing_prompt(self):
        with pytest.raises(ValueError, match="prompt"):
            GenerationRequest.from_dict({"max_new_tokens": 8})

    def test_response_to_dict(self):
        resp = GenerationResponse(
            text="回答", model="m", backend="transformers", usage={"total_tokens": 3}
        )
        data = resp.to_dict()
        assert data["text"] == "回答"
        assert data["usage"] == {"total_tokens": 3}


class TestCreateApp:
    """FastAPI 应用（需要 fastapi 与一个假后端）"""

    @pytest.fixture
    def app(self):
        fastapi = pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        backend = _FakeBackend()
        application = create_app(DeployConfig(model_path="m"), backend=backend)
        return TestClient(application), backend

    def test_health(self, app):
        client, _ = app
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["backend"] == "fake"

    def test_info(self, app):
        client, _ = app
        data = client.get("/v1/info").json()
        assert data["model"] == "m"
        assert data["max_new_tokens"] == 256

    def test_generate(self, app):
        client, backend = app
        resp = client.post("/v1/generate", json={"prompt": "你好", "max_new_tokens": 8})
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "fake:你好"
        assert body["usage"]["completion_tokens"] == 1
        assert backend.calls

    def test_generate_uses_config_defaults(self, app):
        client, backend = app
        client.post("/v1/generate", json={"prompt": "hi"})
        assert backend.calls[0].max_new_tokens is None  # 由后端回落到配置默认值


class TestMissingDependencyHints:
    """缺依赖时给出可执行的安装提示"""

    def test_create_app_without_fastapi(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fastapi":
                raise ImportError("no fastapi")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pip install fastapi uvicorn"):
            create_app(DeployConfig(model_path="m"), backend=_FakeBackend())

    def test_run_server_without_uvicorn(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "uvicorn":
                raise ImportError("no uvicorn")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pip install uvicorn"):
            run_server(DeployConfig(model_path="m"), backend=_FakeBackend())


class _FakeBackend(InferenceBackend):
    """测试用后端，不加载任何真实模型"""

    name = "fake"

    def __init__(self):
        self.calls = []

    def load(self, config):
        self.config = config

    def generate(self, request):
        self.calls.append(request)
        return GenerationResponse(
            text=f"fake:{request.prompt}",
            model=self.config.model_path if hasattr(self, "config") else "",
            backend=self.name,
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )
