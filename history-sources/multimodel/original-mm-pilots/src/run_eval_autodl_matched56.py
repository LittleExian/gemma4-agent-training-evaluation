#!/usr/bin/env python3
"""Official-harness runner. Never exposes reference/test patches to the agent."""
import argparse, asyncio, dataclasses, hashlib, json, os, time
from pathlib import Path
from contextvars import ContextVar
CURRENT_TASK=ContextVar("current_task",default=None)
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("DOCKER_HOST", "unix://"+str(ROOT/"docker.sock"))
os.environ.setdefault("OTEL_SDK_DISABLED","true")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP","True")
os.environ.setdefault("LITELLM_TELEMETRY","False")
os.environ.setdefault("OPENAI_API_KEY","EMPTY")
import yaml
from adk_submission import ModelRegistry, discover_adapters
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.models import load_tasks
from swegemma.evaluate import Evaluator, resolve_task_snapshot_paths
from swegemma.harness.verification import verify_task
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig

async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["gold","empty","agent"],required=True)
    ap.add_argument("--ids",nargs="*")
    ap.add_argument("--wheels",default="wheels")
    ap.add_argument("--submission",default="public_flexonafft")
    ap.add_argument("--run",required=True)
    ap.add_argument("--image",default="swebench-sandbox:official-20261013")
    args=ap.parse_args()
    out=ROOT/"runs"/args.run
    out.mkdir(parents=True,exist_ok=True)
    tasks=load_tasks(ROOT/"data/tasks.jsonl")
    if args.ids:
        wanted=set(args.ids);tasks=[t for t in tasks if t.instance_id in wanted]
        assert len(tasks)==len(wanted),"Unknown/duplicate task IDs"
    submission=ROOT/"submissions"/args.submission
    raw=yaml.safe_load((submission/"eval_config.yaml").read_text()).get("evaluation",{})
    models=ModelRegistry()
    if args.mode=="agent":
        import litellm
        from google.adk.models.lite_llm import LiteLlm
        from adk_submission.resolvers.generation import install_litellm_reasoning_patch
        install_litellm_reasoning_patch()
        litellm.drop_params=True
        from litellm.integrations.custom_logger import CustomLogger
        class RequestRecorder(CustomLogger):
            async def async_log_success_event(self,kwargs,response_obj,start_time,end_time):
                # Preserve exact model-visible conversation for later successful-trace SFT.
                # Do not record credentials, headers, environment, gold patches, or test patches.
                params=kwargs.get("optional_params",{})
                response=response_obj.model_dump(mode="json")
                record={"task_id":CURRENT_TASK.get(),"start":str(start_time),"end":str(end_time),
                    "messages":kwargs.get("messages",[]),"tools":params.get("tools",[]),
                    "sampling":{k:params.get(k,kwargs.get(k)) for k in ["temperature","top_p","max_tokens","max_completion_tokens","seed","reasoning_effort","thinking_token_budget","chat_template_kwargs","extra_body"]},
                    "response":{k:response.get(k) for k in ["id","model","choices","usage"]}}
                with (out/"llm_requests.jsonl").open("a") as log:
                    log.write(json.dumps(record,ensure_ascii=False,default=str)+"\n")
        litellm.callbacks=[RequestRecorder()]
        model="gemma-4-31b-it-qat-w4a16-ct"
        models.register(model,LiteLlm(model="openai/"+model,api_base="http://127.0.0.1:8106/v1",api_key="EMPTY",num_retries=3))
        for adapter_name in discover_adapters(submission).adapters:
            models.register("adapter:"+adapter_name,LiteLlm(model="openai/"+adapter_name,api_base="http://127.0.0.1:8106/v1",api_key="EMPTY",num_retries=3))
    cfg=EvalConfig(
        tasks_path=ROOT/"data/tasks.jsonl",snapshots_dir=ROOT/"data/matched56_snapshots",
        results_dir=out,submission_dir=submission,models=models,
        image=args.image,sandbox="subprocess",wheels_dir=ROOT/"data"/args.wheels,
        graph_dir=str(ROOT/"data/graphs"),embeddings_dir=str(ROOT/"data/embeddings"),
        timeout_seconds=raw.get("timeout_seconds",240),verification_timeout_seconds=300,
        max_time_minutes=raw.get("max_time_minutes",8),max_tool_calls=raw.get("max_tool_calls",28),
        max_turns=raw.get("max_turns",80),verbose=False,display_mode="quiet",
        context_cache_config=ContextCacheConfig(min_tokens=2048,ttl_seconds=1800,cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=15,overlap_size=2,token_threshold=14336,event_retention_size=5),
    )
    # Official cache uses a constant sp_base.tar name; isolate by wheel manifest.
    import tempfile
    wheel_manifest="\n".join(f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in sorted(cfg.wheels_dir.glob("*.whl")))
    wheel_key=hashlib.sha256(wheel_manifest.encode()).hexdigest()[:16]
    isolated_tmp=ROOT/"tmp"/("harness_parallel_"+wheel_key+"_"+hashlib.sha256(args.run.encode()).hexdigest()[:12])
    isolated_tmp.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(isolated_tmp)
    os.environ["TMPDIR"]=str(isolated_tmp)
    import swegemma.evaluate as evaluate_module
    from gemmacodeagent_mario_adapter import ChrootManager
    evaluate_module.SubprocessManager=ChrootManager
    evaluator=Evaluator(cfg)
    resultfile=out/"task_results.jsonl"
    done=set()
    if resultfile.exists():
        for line in resultfile.read_text().splitlines():
            r=json.loads(line);done.add(r.get("task_id",r.get("instance_id")))
    manifest={"mode":args.mode,"ids":[t.instance_id for t in tasks],"image":args.image,
              "submission":args.submission,"started_unix":time.time(),"verification_timeout_seconds":300,
              "wheel_fingerprint":wheel_key,"wheels_dir":str(cfg.wheels_dir),
              "submission_files":{str(f.relative_to(submission)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(submission.rglob("*")) if f.is_file()},
              "tasks_sha256":hashlib.sha256(cfg.tasks_path.read_bytes()).hexdigest(),
              "compaction_threshold":14336,"execution_concurrency":int(os.environ.get("ADP_CONCURRENCY","1")),"hardware":os.environ.get("GEMMA_RUNTIME_PROFILE","pro6000")+"; TP1 local Landlock/profile adaptation; scorer TP4 L4 timing differs"}
    old_manifest=out/"manifest.json"
    if old_manifest.exists() and done:
        old=json.loads(old_manifest.read_text())
        for key in ("mode","ids","image","submission","wheel_fingerprint","submission_files","tasks_sha256"):
            if key in old and old[key]!=manifest[key]:
                raise RuntimeError(f"Refusing to combine different experiment settings: {key}")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    try:
        for i,t in enumerate(tasks,1):
            if t.instance_id in done: continue
            CURRENT_TASK.set(t.instance_id)
            started=time.perf_counter()
            try:
                if args.mode=="agent":
                    result=await evaluator.evaluate_task(t,task_index=i,total_tasks=len(tasks))
                else:
                    snapshot,base,patch=resolve_task_snapshot_paths(cfg.snapshots_dir,t.instance_id,t.repo)
                    result=await verify_task(evaluator.docker,cfg,t,snapshot,
                        base_snapshot_path=base,patch_path=patch,
                        agent_patch=t.patch if args.mode=="gold" else "",start_time=started)
                row=result.model_dump(mode="json",exclude={"trace"})
            except Exception as e:
                import traceback
                row={"task_id":t.instance_id,"repo":t.repo,"resolved":False,"error_message":str(e),"traceback":traceback.format_exc()}
            row.update(mode=args.mode,wall_seconds=time.perf_counter()-started)
            with resultfile.open("a") as f:
                f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
            print(json.dumps({k:row.get(k) for k in ["task_id","resolved","test_exit_code","error_message","wall_seconds"]}),flush=True)
    finally:
        evaluator.docker.cleanup_all()
    rows=[json.loads(l) for l in resultfile.read_text().splitlines()]
    summary={"n":len(rows),"resolved":sum(bool(r.get("resolved")) for r in rows)}
    summary["resolution_rate"]=summary["resolved"]/max(1,summary["n"])
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(summary,flush=True)
if __name__=="__main__":
    from global_eval_slots import Slot
    with Slot(): asyncio.run(main())
