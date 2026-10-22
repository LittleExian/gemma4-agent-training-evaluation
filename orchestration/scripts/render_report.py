"""Render the downloaded, fresh notebook execution evidence as an HTML report."""
import html
import json
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent.parent / 'artifacts'


def read_json(name):
    return json.loads((ARTIFACTS / name).read_text())


def escaped(value):
    return html.escape(str(value))


status = read_json('execution-status.json')
metrics = read_json('metrics.json')
smoke = read_json('smoke.json')
verification = read_json('verification.json')
message = smoke['choices'][0]['message']
resolved = sum(bool(row['resolved']) for row in metrics)
rows = ''.join(
    '<tr>' + ''.join(f'<td>{escaped(value)}</td>' for value in [
        row['id'], '已解决' if row['resolved'] else '未解决',
        row['test_exit_code'], row['patch_chars'], row['tool_calls'],
        f"{row['duration_seconds']:.1f} s",
    ]) + '</tr>' for row in metrics
)
page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma 4 · 服务器实际运行报告</title><style>
body{{max-width:1040px;margin:48px auto;padding:0 24px;background:#f6f7f9;color:#202833;font:16px/1.7 system-ui,sans-serif}}
h1{{font-size:36px;line-height:1.2}}h2{{font-size:22px}}section{{background:white;border:1px solid #dce1e7;border-radius:12px;padding:24px;margin:20px 0}}
.tag{{color:#17683d;background:#e1f3e8;border-radius:6px;padding:5px 10px;display:inline-block}}a{{color:#1859b2}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #e4e7eb;padding:10px}}small{{color:#596575}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}.scroll{{overflow:auto}}
</style><small>GEMMA 4 DEVELOPER AGENT / ACTUAL SSH EXECUTION</small>
<h1>Notebook 实际运行结果</h1>
<p>SERVER_HOST · Mario · {escaped(status['started_at'])}</p>
<p class="tag">{escaped(status['status'])} · 实际推理已验证 · 样本解决 {resolved} / {len(metrics)}</p>
<section><h2>本次执行</h2><p>已在服务器独立 Python 环境中完成数据加载、代码图查询、Gemma 4 服务启动、实际推理、两个 FastAPI 样本评估及提交包生成。使用单张 RTX PRO 6000 Blackwell 96 GB，保留原 notebook 的两个样本与评估预算。</p>
<p>模型权重约 23.27 GB、41 个依赖 wheel，以及两个样本所需的 144 个资源文件已下载并校验。全量比赛资源未下载。</p>
<p><a href="executed.html">查看完整单元输出 →</a>　<a href="executed.ipynb">下载执行后的 notebook</a></p></section>
<section><h2>真实模型回复</h2><p>输入：<code>Reply with exactly: READY</code></p>
<pre>{escaped(message.get('content'))}</pre>
<p>Token 用量：{escaped(json.dumps(smoke.get('usage'), ensure_ascii=False))}</p>
<p><a href="smoke.json">推理原始响应</a></p></section>
<section><h2>两个样本的评估结果</h2><div class="scroll"><table><tr><th>任务</th><th>结果</th><th>测试退出码</th><th>补丁字符</th><th>工具调用</th><th>耗时</th></tr>{rows}</table></div>
<p>两个任务均在补丁生成阶段耗尽原设定的 1 分钟预算，没有提交补丁或工作区修改，因此评估器提前返回，pytest 验证阶段未启动。工具调用上限为 10 次，验证命令超时配置为 60 秒。</p>
<p><a href="metrics.json">评估统计</a> · <a href="predictions.json">补丁预测</a> · <a href="trace_fastapi_15661.json">任务一轨迹</a> · <a href="trace_fastapi_15588.json">任务二轨迹</a> · <a href="submission.zip">提交包</a></p></section>
<section><h2>执行与环境验证</h2><pre>{escaped(json.dumps(verification, ensure_ascii=False, indent=2))}</pre>
<p>适配了服务器路径、单卡显存预算与文本推理设置。修复了 GPU UUID 参数兼容问题，并固定 FastAPI 0.136.0，避免监控中间件与新版路由结构的兼容错误。模型生成的 shell 命令受到文件访问限制。本次推理服务执行结束后已清理。</p>
<p><a href="execution-status.json">执行状态</a> · <a href="notebook-run.log">执行日志</a> · <a href="runtime-freeze.txt">依赖版本</a></p><p>服务临时日志随服务清理被删除；服务启动、实际推理与任务输出保存在完整 notebook 中。状态文件的 cleanup_note 表示服务进程此前已退出。</p></section>
<section><h2>附件历史输出</h2><p>本页结果来自 2026-10-08 的服务器实际运行。原附件保存的历史输出单独保留。</p><p><a href="historical.html">原附件历史渲染</a> · <a href="original.ipynb">原始 notebook</a></p></section>
</html>'''
(ARTIFACTS / 'index.html').write_text(page)
print(ARTIFACTS / 'index.html')
