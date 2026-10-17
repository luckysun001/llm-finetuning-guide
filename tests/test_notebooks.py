"""Jupyter 教程 notebooks 的静态校验。

不执行 notebook（执行需要 torch / GPU / 网络），只做三件事：

1. JSON 结构合法、nbformat 版本正确
2. 每个代码单元的源码能 ``compile()``（语法不过关会立刻暴露）
3. 教程引用的公开 API 确实存在于 ``llm_finetuning`` 中

这样 notebooks 不会因为重构 API 而静默腐化 —— 以前这类问题只有在
用户照着教程跑、报错时才会被发现。
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

# 教程里出现的 shell 命令与魔术命令不需要参与编译
_SKIP_PREFIXES = ("%", "!", "?")
# 这些名字在 notebook 中作为字符串/注释出现，不参与 API 存在性检查
_KNOWN_NON_API = {
    "sys",
    "json",
    "math",
    "torch",
    "matplotlib",
    "plt",
    "importlib",
    "Path",
}


def _notebook_paths():
    if not NOTEBOOK_DIR.is_dir():
        return []
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


ALL_NOTEBOOKS = _notebook_paths()


@pytest.mark.skipif(not ALL_NOTEBOOKS, reason="没有 notebooks/ 目录")
def test_notebooks_exist():
    assert len(ALL_NOTEBOOKS) >= 5, f"教程数量不足: {[p.name for p in ALL_NOTEBOOKS]}"


@pytest.mark.skipif(not ALL_NOTEBOOKS, reason="没有 notebooks/ 目录")
@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] >= 4
    assert isinstance(nb["cells"], list) and nb["cells"]
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code", "raw")
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert "outputs" in cell


@pytest.mark.skipif(not ALL_NOTEBOOKS, reason="没有 notebooks/ 目录")
@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_are_syntactically_valid(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]).strip()
        if not source or source.startswith(_SKIP_PREFIXES):
            continue
        try:
            compile(source, f"{path.name}#cell{idx}", "exec")
        except SyntaxError as exc:  # pragma: no cover - 仅在教程写错时触发
            pytest.fail(f"{path.name} 第 {idx} 个代码单元语法错误: {exc}")


@pytest.mark.skipif(not ALL_NOTEBOOKS, reason="没有 notebooks/ 目录")
@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_imported_symbols_exist(path):
    """检查 `from llm_finetuning... import X` 中的 X 是否真实存在。

    只做静态解析（不执行导入），避免把测试拖进重型依赖。
    """
    import llm_finetuning
    import llm_finetuning.data
    import llm_finetuning.deploy
    import llm_finetuning.export
    import llm_finetuning.methods
    import llm_finetuning.tuning

    modules = {
        "llm_finetuning": llm_finetuning,
        "llm_finetuning.data": llm_finetuning.data,
        "llm_finetuning.deploy": llm_finetuning.deploy,
        "llm_finetuning.export": llm_finetuning.export,
        "llm_finetuning.methods": llm_finetuning.methods,
        "llm_finetuning.tuning": llm_finetuning.tuning,
    }

    nb = json.loads(path.read_text(encoding="utf-8"))
    pattern = re.compile(r"from\s+(llm_finetuning[\w.]*)\s+import\s+([^\n]+)")

    checked = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for module_name, symbols in pattern.findall(source):
            mod = modules.get(module_name)
            assert mod is not None, f"{path.name}: 未知模块 {module_name}"
            # 去掉括号与换行续接，提取纯符号名
            names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", symbols.split("#")[0])
            for name in names:
                if name in _KNOWN_NON_API or name == "import":
                    continue
                assert hasattr(mod, name), (
                    f"{path.name}: {module_name} 中没有 {name} —— 教程引用了不存在的 API"
                )
                checked += 1

    assert checked > 0, f"{path.name} 未检出任何 API 引用，正则可能失效"
