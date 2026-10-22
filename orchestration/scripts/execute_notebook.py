"""Adapt and execute the uploaded notebook, saving fresh cell outputs."""
import copy
import json
import os
import signal
import socket
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
original = nbformat.read(ROOT / "original.ipynb", as_version=4)
notebook = copy.deepcopy(original)
notebook.metadata.pop("papermill", None)
notebook.metadata.pop("kaggle", None)
notebook.metadata.kernelspec = {"display_name": "Gemma 4 server runtime", "language": "python", "name": "gemma4"}
for cell in notebook.cells:
    cell.metadata = {}
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
notebook.cells[2].source = '''import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path
ROOT = Path(os.environ['GEMMA_RUN_ROOT'])
for key, value in {
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
    'TRANSFORMERS_NO_TF': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1200',
    'VLLM_NO_USAGE_STATS': '1',
    'OTEL_SDK_DISABLED': 'true',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
}.items():
    os.environ[key] = value
WHEELHOUSE_DIR = ROOT / 'wheelhouse'
print('Isolated Python:', sys.executable)
for package in ['torch', 'vllm', 'swegemma', 'adk-submission', 'transformers', 'xgrammar', 'google-adk', 'fastapi', 'prometheus-fastapi-instrumentator']:
    print(package, importlib.metadata.version(package))
import torch
print('Selected CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('Visible CUDA devices:', torch.cuda.device_count(), torch.cuda.get_device_name(0))
print('Actual GPU UUID:', str(torch.cuda.get_device_properties(0).uuid))
assert torch.cuda.device_count() == 1
print('Dependencies were installed in the dedicated runtime before notebook execution.')
'''
notebook.cells[4].source = notebook.cells[4].source.replace(
    "Path('/kaggle/input/competitions/gemma-4-developer-agent')", "ROOT / 'data'"
).replace("Path('/kaggle/working')", "ROOT / 'work'")
source = notebook.cells[8].source
source = source.replace("Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')", "ROOT / 'model'")
source = source.replace("port=8000,", "port=int(os.environ['GEMMA_PORT']),")
source = source.replace("gpu_memory_utilization=0.90,", "gpu_memory_utilization=float(os.environ.get('GEMMA_GPU_MEMORY_FRACTION', '0.33')),\n    extra_args=['--max-num-seqs', '4', '--limit-mm-per-prompt', '{\"image\":0,\"audio\":0,\"video\":0}', '--enforce-eager'],")
source = source.replace("server_instance.start()", "print('Server log:', server_instance.log_path, flush=True)\n(ROOT / 'work/server-log-path.txt').write_text(server_instance.log_path)\nserver_instance.start()\n(ROOT / 'work/server.pid').write_text(str(server_instance.process.pid))")
notebook.cells[8].source = source
server_setup = source.index('server_instance.start()')
source = source[:server_setup] + 'try:\n' + '\n'.join('    ' + line if line else '' for line in source[server_setup:].splitlines()) + '\nexcept BaseException:\n    server_instance.stop()\n    raise\n'
notebook.cells[8].source = source
evaluation = notebook.cells[10].source
confinement = '''# Apply filesystem confinement to commands executed by the subprocess backend.
import swegemma.sandbox.subprocess as subprocess_backend
_original_execute_command = subprocess_backend.execute_subprocess_command
def confined_execute_command(command, cwd, env, timeout=300, **kwargs):
    workspace = Path(os.path.abspath(cwd))
    sandbox_root = workspace.parent
    if (sandbox_root.is_symlink() or workspace.is_symlink()
            or sandbox_root.resolve().parent != (ROOT / 'tmp').resolve()
            or not sandbox_root.name.startswith('swegemma_sandbox_')):
        raise RuntimeError('Unexpected subprocess sandbox root')
    launcher = [sys.executable, '-I', str(ROOT / 'landlock_guard.py')]
    for path in ['/usr', '/etc', '/opt/anaconda3', str(ROOT / 'runtime'),
                 str(ROOT / 'landlock_guard.py'), '/sys', '/proc/cpuinfo',
                 '/proc/meminfo', '/proc/stat', '/proc/sys', '/dev/urandom', '/dev/random']:
        launcher.extend(['--read', path])
    launcher.extend(['--write', str(sandbox_root), '--write', '/dev/null', '--'])
    shell_command = ['/bin/bash', '-c', command] if isinstance(command, str) else list(command)
    return _original_execute_command(command=launcher + shell_command, cwd=cwd,
        env=env, timeout=timeout, shell=False,
        max_output_chars=kwargs.get('max_output_chars', 10 * 1024 * 1024))
subprocess_backend.execute_subprocess_command = confined_execute_command
print('Subprocess command filesystem confinement enabled.')
'''
evaluation = confinement + evaluation
evaluation = evaluation.replace("predictions = []", "predictions = []\nmetrics = []")
evaluation = evaluation.replace("    predictions.append", "    metrics.append({'id': task.instance_id, 'resolved': result.resolved, 'test_exit_code': result.test_exit_code, 'patch_chars': len(result.agent_patch or ''), 'tool_calls': result.tool_calls, 'duration_seconds': result.duration_seconds})\n    (WORKING_DIR / 'metrics.json').write_text(__import__('json').dumps(metrics, indent=2))\n    predictions.append")
evaluation = evaluation.replace("display(submission_df)", "display(submission_df)\nsubmission_df.to_json(WORKING_DIR / 'predictions.json', orient='records')")
notebook.cells[10].source = "try:\n" + "\n".join("    " + line if line else "" for line in evaluation.splitlines()) + "\nfinally:\n    server_instance.stop()\n    print('Stopped this notebook inference server.')\n"
smoke = nbformat.v4.new_code_cell('''import requests
import json
try:
    response = requests.get(server_instance.base_url + '/models', timeout=30)
    response.raise_for_status()
    served_model = response.json()['data'][0]['id']
    reply = requests.post(server_instance.base_url + '/chat/completions', json={
        'model': served_model,
        'messages': [{'role': 'user', 'content': 'Reply with exactly: READY'}],
        'max_tokens': 64,
        'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False},
    }, timeout=120)
    reply.raise_for_status()
    smoke_result = reply.json()
    (ROOT / 'work/smoke.json').write_text(json.dumps(smoke_result, indent=2))
    print('Served model:', served_model)
    print('Model smoke response:', smoke_result['choices'][0]['message'])
    print('Token usage:', smoke_result.get('usage'))
except BaseException:
    server_instance.stop()
    raise
''')
notebook.cells.insert(9, smoke)
notebook.cells.insert(0, nbformat.v4.new_markdown_cell(
    f"# 本次服务器真实执行\n\n时间：{stamp}（Asia/Shanghai）。"
    "在独立 Python 环境中运行，使用单张 RTX PRO 6000，路径适配到任务目录。"
    "保留原两个样本及评估预算；增加真实推理检查和运行统计。所有附件历史输出已清空。"
))
os.environ['GEMMA_RUN_ROOT'] = str(ROOT)
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    os.environ['GEMMA_PORT'] = str(sock.getsockname()[1])
status = {"status": "running", "started_at": stamp, "last_cell": None}
(WORK / 'server.pid').unlink(missing_ok=True)


def save(**kwargs):
    nbformat.write(notebook, WORK / "executed.ipynb")
    (WORK / "execution-status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))


def cell_start(cell_index, **kwargs):
    status['last_cell'] = cell_index
    print(f"CELL START {cell_index}", flush=True)
    save()


def cell_done(cell_index, **kwargs):
    print(f"CELL DONE {cell_index}", flush=True)
    save()


client = NotebookClient(notebook, timeout=2400, kernel_name="gemma4",
    resources={"metadata": {"path": str(WORK)}},
    on_cell_start=cell_start, on_cell_executed=cell_done, on_cell_error=save)
save()
try:
    client.execute()
    status['status'] = 'completed'
except BaseException as exc:
    status['status'] = 'failed'
    status['error'] = f"{type(exc).__name__}: {exc}"
    print(status['error'], flush=True)
finally:
    # Only clean up the process recorded by this run, and check its command first.
    pid_path = WORK / 'server.pid'
    if pid_path.exists():
        pid = int(pid_path.read_text())
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'vllm' in command and str(ROOT / 'model').encode() in command:
                import psutil
                process = psutil.Process(pid)
                for child in process.children(recursive=True):
                    child.terminate()
                process.terminate()
        except Exception as exc:
            status['cleanup_note'] = f'{type(exc).__name__}: {exc}'
    status['finished_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    save()
    body, _ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
    (WORK / 'executed.html').write_text(body, encoding='utf-8')
print(json.dumps(status, ensure_ascii=False), flush=True)
raise SystemExit(0 if status['status'] == 'completed' else 1)
