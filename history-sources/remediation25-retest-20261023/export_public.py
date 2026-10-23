"""Read-only closed-set evaluation exporter. Supply root and SSH host at execution time."""
import sys, os, json, pathlib, hashlib, subprocess, collections, csv, re
from datetime import datetime, timezone

def digest(b): return hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()
def select(v,keys): return {k:v[k] for k in keys if k in v}
def collect(root):
    root=pathlib.Path(root);run=root/'runs/full129-gpu0-gpu2';host=os.getenv('PUBLICATION_SSH_HOST','')
    def safe(v):
        if isinstance(v,str):
            v=v.replace(str(root.parent),'/workspace').replace('/home/'+root.parent.name,'/home/runner')
            return v.replace(host,'SERVER_HOST') if host else v
        if isinstance(v,dict): return {safe(k):safe(x) for k,x in v.items()}
        if isinstance(v,list): return [safe(x) for x in v]
        return v
    def load(p): return json.loads(p.read_text())
    files={};sources={};metadata=[];rows=[];fingerprints=[];traces=[];profiles={}
    def record(p):
        metadata.append({'path':str(p.relative_to(root)),'original_sha256':sha(p),'original_bytes':p.stat().st_size})
    def put(name,v): files[name]=safe(v)
    protocol=load(run/'protocol.json')
    for name in ['protocol.json','controller-status.json','manifest.json','progress.json','summary.json']:
        p=run/name;record(p);put(name,load(p))
    for name in ['initial-request-verification.json','served-route-effect-gpu0.json','served-route-effect-gpu2.json']:
        p=root/'reports'/name;record(p);put(name,load(p))
    p=root/'reports/control-smoke-verification.json';j=load(p);record(p)
    put('control-smoke-verification.json',select(j,['passed','task_id','gold_resolved','empty_resolved','effective_budget_asserted']))
    for gpu in [0,2]:
        p=root/f'reports/runtime-verification-gpu{gpu}.json';record(p);j=load(p)
        put(f'runtime-verification-gpu{gpu}.json',{k:v for k,v in j.items() if k!='evidence'})
        p=root/f'services/gpu{gpu}/health.json';record(p);j=load(p)
        put(f'health-gpu{gpu}.json',select(j,['passed','concurrency','models','thinking_tokens','seconds','tokens_per_second','finished']))
        p=root/f'services/gpu{gpu}/process.json';record(p);j=load(p)
        put(f'service-gpu{gpu}.json',select(j,['arm','pid','create_time','physical_gpu_index_at_start','port','command','memory_utilization','max_num_seqs','adapter_tensor_count','adapter_dtype','adapter_sha256','adapter_config_sha256','base_identity','checkpoint_source','wrapper_sha256','base_generation_config_sha256','effective_top_k','seed','cuda_graph_requested','requested_content_format','adapter_runtime_dtype','adapter_file_transformation','checkpoint_complete_sha256','adapter_audit_sha256','tokenizer_path','tokenizer_sha256','training_serialization','evaluation_serialization','resolved_content_format','wrapper_started_unix']))
    errorcounts=collections.Counter()
    for f in sorted((run/'results').glob('*.json')):
        j=load(f);record(f);a=j.get('agent') or {};c=j.get('controls') or {};iid=j['instance_id']
        row=select(j,['instance_id','normal_output','resolved','status','started_unix','finished_unix','required_profile_ids'])
        row.update({'source_result_sha256':sha(f),'eligibility':c.get('gate','excluded' if j['status']=='preflight_excluded' else 'unknown'),'exclusion_reasons':c.get('reasons',[j['reason']] if j.get('reason') else []),'repo':c.get('repo'),'reference_control_counts':select(c,['gold_counts','empty_counts','empty_raw_exit_code','profile_freeze_sha256','record_fingerprint','rule_version'])})
        row['agent']=select(a,['status','duration_seconds','wall_seconds','total_llm_calls','tool_calls','patch_chars','agent_timed_out','error_code','infra_error','test_exit_code','fingerprint','result_sha256'])
        em=a.get('error_message') or '';e=em.lower()
        cat='session_timeout' if a.get('agent_timed_out') else 'none' if not e else 'tool_budget' if 'tool' in e and any(x in e for x in ['budget','limit','maximum']) else 'context_overflow' if any(x in e for x in ['context length','context_length','maximum context','max_model_len','input tokens']) else 'other'
        row['agent'].update({'error_category':cat,'error_message_sha256':digest(em.encode()) if em else None});errorcounts[cat]+=1
        row['validations']=[select(x,['resolved','status','test_exit_code','infra_error','agent_timed_out','error_code']) for x in j.get('validations',[])]
        for p in (run/'tasks'/f.stem).glob('agent-*'):
            if not p.is_dir():continue
            inv=load(p/'invocation.json');assert inv.get('patch_file') is None
            fp=load(p/'fingerprint.json');inp=fp['inputs'];pr=inp.get('profile',{});pid=pr.get('profile_id')
            if pid:profiles[pid]=select(pr,['family','python_version','requirements','fingerprint','profile_id','actual_freeze','interpreter_version'])
            item={'instance_id':iid,'fingerprint':fp['fingerprint'],'fingerprint_original_sha256':sha(p/'fingerprint.json'),'invocation_original_sha256':sha(p/'invocation.json'),'source_directory':str(p.relative_to(run)),'invocation':select(inv,['task_id','profile_id','arm','endpoint','patch_file']),'inputs':{k:v for k,v in inp.items() if k!='profile'}}
            req=p/'llm_requests.jsonl'
            if req.is_file():
                with req.open() as rf:first=rf.readline()
                item['first_request_sampling']=json.loads(first)['sampling'] if first else None
            fingerprints.append(item)
            patch=p/'agent.patch'
            if patch.is_file():
                assert a.get('mode')=='agent';text=patch.read_text();target='generated-patches/'+iid+'.patch';files[target]=safe(text)
                row['generated_patch']={'path':target,'original_sha256':sha(patch),'original_bytes':patch.stat().st_size,'source_mode':'agent','supplied_patch':False,'invocation_original_sha256':sha(p/'invocation.json')}
            for q in [*list((p/'traces').glob('*.json')),req]:
                if q.is_file():traces.append({'path':str(q.relative_to(run)),'sha256':sha(q),'bytes':q.stat().st_size,'contents_published':False})
        rows.append(row)
    assert len(rows)==len({r['instance_id'] for r in rows})==129
    assert {r['instance_id'] for r in rows}==set(files['manifest.json']['task_ids'])
    assert sum(bool(r['resolved']) for r in rows)==54
    assert len(fingerprints)==126 and sum(k.startswith('generated-patches/') for k in files)==126
    for name in ['gemmacodeagent_step25_retest_server.py','gemmacodeagent_step25_retest_task.py','gemmacodeagent_step25_retest_controller.py','test_gemmacodeagent_step25_retest_server.py','test_gemmacodeagent_step25_retest_task.py','test_gemmacodeagent_step25_retest_controller.py','budget_contract.py','gemmacodeagent_morning25_server.py','gemmacodeagent_morning25_task.py','gemmacodeagent_morning25_controller.py','gemmacodeagent_mm_task.py','gemmacodeagent_mm_run.py','gemmacodeagent_full_task.py','gemmacodeagent_full_adapter.py','gemmacodeagent_native_guard.py','gemmacodeagent_mario_adapter.py','run_eval_autodl_matched56.py','gemmacodeagent_owned_cleanup.py']:
        p=root/'src'/name;expected=protocol['source_sha256'].get('src/'+name)
        sources['src/'+name]={'original_sha256':sha(p),'recorded_execution_sha256':expected,'matches_recorded_execution':sha(p)==expected,'text':safe(p.read_text())}
    p=root/'reports/source-at-gpu0-launch/gemmacodeagent_step25_retest_server.py'
    sources['gpu0-launch/gemmacodeagent_step25_retest_server.py']={'original_sha256':sha(p),'recorded_execution_sha256':files['service-gpu0.json']['wrapper_sha256'],'matches_recorded_execution':sha(p)==files['service-gpu0.json']['wrapper_sha256'],'text':safe(p.read_text())}
    for name in ['agent.yaml','eval_config.yaml','configs/sampling.yaml']:
        p=root/'submissions/mm15_6000'/name;sources['submission/'+name]={'original_sha256':sha(p),'text':safe(p.read_text())}
    prompt=root/'submissions/mm15_6000/prompts/system.md';pt=prompt.read_text()
    eligible=[r for r in rows if r['eligibility']=='eligible'];summary={'run_id':'remediation25-retest-20261023','scope':'full129','completion':'completed','actual_unique_tasks':129,'total':129,'eligible':len(eligible),'excluded':129-len(eligible),'excluded_ids':[r['instance_id'] for r in rows if r['eligibility']!='eligible'],'resolved':54,'agent_timeouts':sum(r['status']=='agent_timeout' for r in rows),'infra_error_results':sum(bool(r['agent'].get('infra_error')) for r in rows),'normal_outputs':sum(bool(r['normal_output']) for r in rows),'statuses':dict(collections.Counter(r['status'] for r in rows)),'agent_error_categories':dict(errorcounts),'physical_gpu_ids':[0,2],'workers_per_gpu':2,'total_workers':4,'service_max_num_seqs':4,'model':'google/gemma-4-31B-it-qat-w4a16-ct','checkpoint':25,'checkpoint_family':'remediation-20261022 (not prefix-v2 training checkpoint)','adapter_sha256':protocol['adapter_sha256'],'runtime_budget':{'minutes':15,'tool_calls':56,'turns':160,'thinking_tokens':6000,'top_k':-1,'seed':42},'prompt_original_sha256':sha(prompt),'prompt_budget_lines':[x for x in pt.splitlines() if x.startswith('## Budget:') or x.startswith('- You have at most')],'dataset_sha256':protocol['dataset_sha256'],'gate_sha256':protocol['gate_sha256'],'cuda_graph_mode':'FULL_AND_PIECEWISE','cuda_graph_capture_verified_both_gpus':all(files[f'runtime-verification-gpu{g}.json']['cuda_graph_captured'] for g in [0,2]),'requested_content_format':'auto','resolved_content_format':'openai','request_proof_files_checked':files['initial-request-verification.json']['request_files_checked'],'comparability_notes':['Same remediation checkpoint-25 adapter bytes as prior morning25 run. This rerun changes sampling, concurrency and serving execution settings, so score difference is not an isolated adapter or training effect.','GPU0 launched using a saved wrapper snapshot; GPU2 used the current wrapper. Each launch hash is separately recorded.','Raw prompts, responses, test output, reference patches, dataset and weights are omitted.']}
    for field,fun in [('first_task_started_unix',min),('last_task_finished_unix',max)]:
        key='started_unix' if field.startswith('first') else 'finished_unix';t=fun(r[key] for r in rows if r.get(key));summary[field]=t;summary[field.replace('_unix','_utc')]=datetime.fromtimestamp(t,timezone.utc).isoformat()
    put('summary.json',summary);put('task-results.json',rows);put('per-task-budget-and-source-hashes.json',fingerprints);put('environment-profiles.json',profiles);put('trace-and-request-inventory.json',{'files':traces,'contents_published':False});put('original-artifact-hashes.json',metadata)
    freeze=subprocess.check_output([str(root/'envs/eval/bin/python'),'-m','pip','freeze'],text=True)
    versions=subprocess.check_output([str(root/'envs/eval/bin/python'),'-m','pip','list','--format=freeze'],text=True)
    put('runtime-dependency-evidence.json',{'package_versions':versions,'python_executable_target':str((root/'envs/eval/bin/python').resolve()),'pip_freeze_original_sha256':digest(freeze.encode()),'pip_freeze':freeze})
    return {'files':files,'sources':sources}

def materialize(data,stage):
    stage=pathlib.Path(stage);dest=stage/'results/remediation25-retest-20261023';src=stage/'history-sources/remediation25-retest-20261023';known={}
    for p in stage.rglob('*'):
        if p.is_file() and '.git' not in p.parts and not p.is_relative_to(dest) and not p.is_relative_to(src) and p.stat().st_size<3_000_000:known.setdefault(sha(p),str(p.relative_to(stage)))
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
    for name,value in data['files'].items():
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(value,str):p.write_text(value)
        else:write(p,value)
    refs=[]
    for name,item in data['sources'].items():
        public=digest(item['text'].encode());entry={k:v for k,v in item.items() if k!='text'};entry.update({'original_path':name,'public_sha256':public})
        if public in known:entry.update({'published_path':known[public],'reused_existing_file':True})
        else:
            p=src/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(item['text']);entry.update({'published_path':str(p.relative_to(stage)),'reused_existing_file':False});known[public]=entry['published_path']
        refs.append(entry)
    write(dest/'source-index.json',{'hash_policy':'original_sha256 is unredacted server bytes; public_sha256 hashes the generalized published copy.','files':refs})
    rows=data['files']['task-results.json'];cols=['instance_id','repo','eligibility','status','normal_output','resolved','started_unix','finished_unix','duration_seconds','tool_calls','total_llm_calls','patch_chars','agent_timed_out','infra_error','test_exit_code','error_category']
    with (dest/'task-results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in rows:w.writerow({k:r[k] if k in r else r.get('agent',{}).get(k) for k in cols})
    for r in rows:
        if r.get('generated_patch'):r['generated_patch']['public_sha256']=sha(dest/r['generated_patch']['path'])
    write(dest/'task-results.json',rows)
    runtime=data['files']['runtime-dependency-evidence.json'];freeze=runtime.pop('pip_freeze');versions=runtime.pop('package_versions');h=digest(freeze.encode())
    existing=stage/'environments/evaluation-runtime-pinned.txt'
    runtime['existing_published_runtime_path']=str(existing.relative_to(stage));runtime['existing_published_runtime_sha256']=sha(existing);runtime['matches_existing_published_runtime']=sha(existing)==h
    if not runtime['matches_existing_published_runtime']:(dest/'evaluation-runtime-pinned.txt').write_text(freeze)
    canonical='\n'.join(sorted(versions.splitlines(),key=lambda x:x.split('==')[0]))+'\n'
    runtime['canonical_versions_original_sha256']=digest(versions.encode())
    runtime['sorted_versions_sha256']=digest(canonical.encode())
    runtime['sorted_versions_matches_existing_runtime']=digest(canonical.encode())==sha(existing)
    runtime['canonical_versions_reuse_path']=str(existing.relative_to(stage))
    runtime['dependency_verification_note']='Package/version lines are compared after lexical package-name sorting; direct-url pip freeze separately records package provenance.'
    write(dest/'runtime-dependency-evidence.json',runtime)
    (dest/'README.md').write_text('# Remediation checkpoint-25 retest, 2026-10-23\n\nCompleted all 129 tasks: 54/126 eligible resolved, 3 agent session timeouts, 3 reference-control exclusions, zero infrastructure error results. Two PRO 6000 GPUs (physical IDs 0 and 2), two workers each.\n\nThe adapter is the remediation checkpoint-25, not a checkpoint from the later prefix-v2 training. Sampling uses top_k=-1, seed=42, thinking budget 6000, 15 minutes, 56 tools, 160 turns. CUDA graph capture and auto→openai content-format resolution are verified for both servers. Initial request proof contains parameters and hashes only. GPU0 launch source snapshot and GPU2/current source are tracked separately.\n\nAll agent patches come from agent-mode invocations without supplied patches. Control patch contents, full task prompts, dataset rows, raw test output, requests/responses/traces, credentials and model weights are excluded. Original raw artifact hashes and sizes are inventories only. Paths are generalized; original and public-copy hashes are distinguished. Source-index references identical files already published in this repository. Runtime dependency evidence records whether the existing pinned runtime has identical bytes.\n')
    for root in [dest,src]:
        for p in root.rglob('*'):
            if p.is_file():
                t=p.read_text();assert not re.search(r'\b(?:hf_[A-Za-z0-9]{20,}|wandb_v1_[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})',t),str(p)
                assert not re.search('/mnt'+'/z1/'+r'[^/\s]+',t),str(p)
    files=[{'path':str(p.relative_to(stage)),'sha256':sha(p),'bytes':p.stat().st_size} for root in [dest,src] for p in root.rglob('*') if p.is_file() and p.name!='PUBLICATION_MANIFEST.json']
    write(dest/'PUBLICATION_MANIFEST.json',{'files':files})
    print(json.dumps({'files':len(files)+1,'patches':sum(x['path'].endswith('.patch') for x in files),'source_refs':len(refs),'source_mismatches':[x['original_path'] for x in refs if x.get('matches_recorded_execution') is False],'runtime_matches_existing':runtime['matches_existing_published_runtime'],'summary':data['files']['summary.json']},indent=2))

if __name__=='__main__':
    if sys.argv[1]=='--remote':print(json.dumps(collect(sys.argv[2]),ensure_ascii=False))
    else:
        host=os.environ['PUBLICATION_SSH_HOST'];root=sys.argv[1];stage=sys.argv[2]
        code=pathlib.Path(__file__).read_bytes()
        remote=subprocess.run(['ssh','-o','BatchMode=yes',host,'python3','-','--remote',root],input=code,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        materialize(json.loads(remote.stdout),stage)
