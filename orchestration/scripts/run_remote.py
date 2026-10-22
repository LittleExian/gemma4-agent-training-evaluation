"""Execute diagnostics and a guarded notebook copy; render all evidence."""
import copy
import html
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbconvert import HTMLExporter


ROOT = Path(__file__).resolve().parent
STAMP = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
original = nbformat.read(ROOT / "original.ipynb", as_version=4)


def render(notebook, stem):
    nbformat.write(notebook, ROOT / f"{stem}.ipynb")
    body, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
    (ROOT / f"{stem}.html").write_text(body, encoding="utf-8")


diagnostic = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_markdown_cell(
        f"# 本次 SSH 环境诊断\n\n运行时间：{STAMP}（Asia/Shanghai）\n\n"
        "目标：SERVER_HOST。以下输出均为本次服务器执行所得。GPU 小矩阵运算仅验证 CUDA，"
        "不代表 Gemma 4 推理已成功。"
    ),
    nbformat.v4.new_code_cell('''import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path
print("HOST:", platform.node())
print("Python:", platform.python_version())
print(subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv"], text=True))
required = {
    "competition": "/kaggle/input/competitions/gemma-4-developer-agent",
    "wheelhouse": "/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse",
    "model": "/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2",
}
print("Required paths:", json.dumps({k: {"path": v, "exists": Path(v).exists()} for k, v in required.items()}, indent=2))
'''),
    nbformat.v4.new_code_cell('''probe = r"""
import importlib.util, json
packages = ['torch','vllm','swegemma','adk_submission','litellm','pandas','yaml']
print(json.dumps({p: importlib.util.find_spec(p) is not None for p in packages}, indent=2))
"""
result = subprocess.run(["/usr/bin/python3", "-c", probe], capture_output=True, text=True, timeout=60)
print(result.stdout)
print(result.stderr)
print("package_probe_exit_code:", result.returncode)
'''),
    nbformat.v4.new_code_cell('''gpu_test = r"""
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA unavailable'
with torch.inference_mode():
    x = torch.eye(128, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    assert torch.equal(x, y)
print('GPU:', torch.cuda.get_device_name(0))
print('128 x 128 CUDA matrix multiplication: PASS')
"""
gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
gpu_index = max((tuple(int(v.strip()) for v in row.split(',')) for row in gpu_rows), key=lambda row: row[1])[0]
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_index)}
print('Selected physical GPU:', gpu_index)
result = subprocess.run(["/usr/bin/python3", "-c", gpu_test], env=env, capture_output=True, text=True, timeout=120)
print(result.stdout)
print(result.stderr)
print('gpu_test_exit_code:', result.returncode)
'''),
])
diagnostic_error = ""
try:
    NotebookClient(diagnostic, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
except Exception as exc:
    diagnostic_error = f"{type(exc).__name__}: {exc}"
finally:
    render(diagnostic, "diagnostics")

attempt = copy.deepcopy(original)
attempt.metadata.pop("papermill", None)
for cell in attempt.cells:
    cell.metadata = {}
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
attempt.cells.insert(0, nbformat.v4.new_markdown_cell(
    f"# 本次 SSH 执行尝试\n\n{STAMP}\n\n所有历史输出已清空。"
    "安装单元前增加资源检查；缺少资源时停止，避免原安装代码修改共享环境。"
    "此副本后续单元保留原始代码，仅供定位；补齐资源后仍需先隔离环境及适配路径。"
))
first_code = next(cell for cell in attempt.cells if cell.cell_type == "code")
first_code.source = '''# SSH preflight added before the original installation cell.
from pathlib import Path
required_paths = [
    Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'),
    Path('/kaggle/input/competitions/gemma-4-developer-agent'),
    Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2'),
]
missing_paths = [str(path) for path in required_paths if not path.is_dir()]
if missing_paths:
    raise FileNotFoundError('Required Kaggle resources are missing: ' + '; '.join(missing_paths))
raise RuntimeError('Shared-server installation requires an isolated environment and path adaptation before proceeding.')
\n''' + first_code.source
status = "unknown"
error = ""
try:
    NotebookClient(attempt, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    status = "completed"
except CellExecutionError as exc:
    status = "blocked_missing_resources" if "FileNotFoundError" in str(exc) else "failed"
    error = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
except Exception as exc:
    status = "execution_error"
    error = f"{type(exc).__name__}: {exc}"
finally:
    render(attempt, "attempted")

historical = copy.deepcopy(original)
historical.cells.insert(0, nbformat.v4.new_markdown_cell(
    "# 附件自带的历史输出 — 非本次 SSH 结果\n\n"
    "原文件记录时间：2026-09-27 UTC。vLLM 启动和打包完成；两个评估任务均超时，"
    "resolved=False，未生成补丁。"
))
render(historical, "historical")
summary = {"time_shanghai": STAMP, "host": "SERVER_HOST", "status": status, "error": error, "diagnostic_error": diagnostic_error}
(ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma 4 · SSH 运行报告</title><style>
body{max-width:1040px;margin:60px auto;padding:0 24px;background:#f6f7f9;color:#202833;font:16px/1.7 system-ui,sans-serif}
h1{font-size:36px;line-height:1.2}h2{font-size:22px}section{background:white;border:1px solid #dce1e7;border-radius:12px;padding:24px;margin:20px 0}
.tag{color:#8b4b04;background:#fff0d9;border-radius:6px;padding:5px 10px;display:inline-block}a{color:#1859b2}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #e4e7eb;padding:10px}code{overflow-wrap:anywhere}small{color:#596575}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
</style><small>GEMMA 4 DEVELOPER AGENT / REMOTE RUN</small><h1>SSH 运行结果</h1>
<p>SERVER_HOST · Mario · __STAMP__</p><p class="tag">连接成功 · 执行停在资源检查 · 尚未运行模型推理</p>
<section><h2>本次实际执行</h2><p>已通过 SSH 运行 Jupyter 环境诊断及 notebook 资源检查。服务器检测到 3 张 RTX PRO 6000 和 1 张 H100；CUDA 实测输出见下方诊断。原 notebook 的三个 Kaggle 输入目录均不存在，因此未进入依赖安装、模型加载或任务评估。</p>
<p><a href="diagnostics.html">查看环境与 CUDA 实测输出 →</a>　<a href="attempted.html">查看本次 notebook 执行与异常 →</a></p></section>
<section><h2>继续运行所需资源</h2><p><a href="https://www.kaggle.com/competitions/gemma-4-developer-agent/data">比赛数据页面</a> · <a href="https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse">Wheelhouse 页面</a></p><table><tr><th>资源</th><th>Notebook 要求</th></tr>
<tr><td>比赛数据</td><td>gemma-4-developer-agent：tasks.jsonl、sample_submission、snapshots、graphs、embeddings、wheels</td></tr>
<tr><td>离线依赖</td><td>metric/gemma-4-developer-agent-wheelhouse</td></tr>
<tr><td>模型权重</td><td>google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2</td></tr></table>
<p>用户已确认资源尚未下载。比赛文件接口返回 HTTP 401（Unauthenticated），需要先配置具有该比赛访问权限的 Kaggle 登录凭据。Wheelhouse 的公开元数据可访问；模型下载尚未验证成功。补齐资源后，需要在独立环境中安装依赖并调整 Kaggle 路径。</p></section>
<section><h2>附件历史结果（非本次运行）</h2><p>原 notebook 记录于 2026-09-27 UTC：vLLM 启动成功，但两项评估均超时，解决数为 0 / 2。</p>
<table><tr><th>任务</th><th>结果</th><th>耗时</th><th>工具调用</th><th>补丁字符</th></tr>
<tr><td>fastapi_15661</td><td>超时 / 未解决</td><td>76.1 s</td><td>4</td><td>0</td></tr>
<tr><td>fastapi_15588</td><td>超时 / 未解决</td><td>73.1 s</td><td>4</td><td>0</td></tr></table>
<p><a href="historical.html">查看原文件完整渲染 →</a></p></section>
<section><h2>输出文件</h2><p><a href="attempted.ipynb">本次尝试 .ipynb</a> · <a href="diagnostics.ipynb">环境诊断 .ipynb</a> · <a href="original.ipynb">原始 notebook</a> · <a href="summary.json">状态 JSON</a></p>
<details><summary>本次异常记录</summary><pre>__ERROR__</pre></details></section></html>'''
if status != "blocked_missing_resources":
    page = page.replace("执行停在资源检查", "执行状态：" + html.escape(status))
    page = page.replace("原 notebook 的三个 Kaggle 输入目录均不存在，因此未进入依赖安装、模型加载或任务评估。", "预检未进入原 notebook 的安装和模型执行代码。具体原因见异常记录。")
if diagnostic_error:
    page = page.replace("本次实际执行</h2>", "本次实际执行</h2><p>环境诊断发生异常，请查看诊断日志。</p>")
(ROOT / "index.html").write_text(page.replace("__STAMP__", html.escape(STAMP)).replace("__ERROR__", html.escape(error + diagnostic_error)), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "error"}, ensure_ascii=False))
