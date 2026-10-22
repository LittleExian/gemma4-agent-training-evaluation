import ast
import tempfile
import unittest
import asyncio
import hashlib
import json
import os
import sys
import types
from unittest.mock import patch
from pathlib import Path
import gemmacodeagent_mm_task as task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SOURCE = PROJECT_ROOT/'references/gemmacodeagent-29984e6-complete/experiments/distributed_model_comparison/run_task.py'
SOURCE = REFERENCE_SOURCE if REFERENCE_SOURCE.is_file() else PROJECT_ROOT/'experiments/distributed_model_comparison/run_task.py'

class MultiModelTaskTests(unittest.TestCase):
    def context(self):
        return dict(base='/data/base', output_dir='/repo/runs/pilot/task', endpoint='http://127.0.0.1:8120/v1', arm='qwen35_swegym', patch_file='/repo/runs/pilot/agent.patch')

    def test_pinned_transform_keeps_official_evaluator_and_private_gold(self):
        result=task.adapt_source(SOURCE.read_text(),self.context())
        ast.parse(result)
        self.assertIn('await evaluator.evaluate_task(t,task_index=i,total_tasks=len(tasks))',result)
        self.assertIn('from gemmacodeagent_full_adapter import ChrootManager',result)
        self.assertIn('agent_patch=t.patch if args.mode=="gold"',result)
        self.assertIn("else (Path('/repo/runs/pilot/agent.patch').read_text() if args.mode==\"verify-patch\"",result)
        self.assertIn('model="gemma-4-31b-it-qat-w4a16-ct"',result)
        self.assertEqual(result.count('model="openai/"+\'qwen35_swegym\''),2)
        self.assertEqual(result.count("'thinking_token_budget': 6000"),2)
        self.assertIn('compaction_interval=15',result)
        self.assertIn('ContextCacheConfig(min_tokens=2048',result)
        self.assertIn('llm_requests.jsonl',result)
        self.assertNotIn('from autodl_chroot',result)

    def test_source_drift_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'source changed'):
            task.adapt_source(SOURCE.read_text()+'\n',self.context())

    def test_loopback_endpoint_and_arm_validation(self):
        self.assertEqual(task.validate_arm('qwen35_swegym'), 'qwen35_swegym')
        for value in ['', '../x', 'foo/bar', 'x?token', 'x\n']:
            with self.assertRaises(ValueError):task.validate_arm(value)
        for endpoint in ['http://outside:8120/v1','http://u:p@127.0.0.1:8120/v1']:
            with self.assertRaises(ValueError):task.validate_endpoint(endpoint)

    def test_submission_budget_validation(self):
        valid={'timeout_seconds':240,'max_time_minutes':15.,'max_tool_calls':56,'max_turns':160}
        task.validate_budget(valid)
        for key in valid:
            with self.assertRaises(ValueError):task.validate_budget({**valid,key:0})

    def test_output_must_be_fresh_and_under_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder)
            output=repo/'runs/pilot/task'
            task.validate_output(repo,output)
            output.mkdir(parents=True)
            (output/'task_results.jsonl').touch()
            with self.assertRaises(ValueError):task.validate_output(repo,output)
            with self.assertRaises(ValueError):task.validate_output(repo,repo/'unsafe')

    def test_budget_timeout_not_infrastructure_and_patch_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            row={'task_id':'rich_1','status':'TIMEOUT','agent_patch':'diff x','test_exit_code':-1}
            result=task.normalize_result(row,output,{},[],'fp','agent')
            self.assertTrue(result['agent_timed_out'])
            self.assertFalse(result['infra_error'])
            self.assertEqual(Path(result['patch_file']).read_text(),'diff x')
            self.assertNotIn('agent_patch',result)

class TaskOrchestrationTests(unittest.TestCase):
    def test_invocation_binds_profile_arm_and_normalized_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            repo=root/'repo';base=root/'base';output=repo/'runs/pilot/task'
            upstream=repo/task.SOURCE_RELATIVE
            upstream.parent.mkdir(parents=True);upstream.write_text(SOURCE.read_text())
            submission=repo/'submissions'/task.SUBMISSION
            submission.mkdir(parents=True);(submission/'eval_config.yaml').write_text('protocol')
            gate=root/'gate.json';gate.write_text('{}')
            profile={'profile_id':'calibrated','environment':str(root/'venv'),'actual_freeze':['bound']}
            def write_json(path,data):
                Path(path).parent.mkdir(parents=True,exist_ok=True)
                Path(path).write_text(json.dumps(data))
            calibration=types.SimpleNamespace(atomic_json=write_json,freeze=lambda _:['bound'],
                sha=lambda path:hashlib.sha256(str(path).encode()).hexdigest())
            adapter=types.SimpleNamespace(profile_candidate=lambda *_:profile,select_profile=lambda *_:profile)
            yaml=types.SimpleNamespace(safe_load=lambda _: {'evaluation':{'timeout_seconds':240,
                'max_time_minutes':15.,'max_tool_calls':56,'max_turns':160}})
            async def evaluate():
                write_json(output/'adapter/commands.json',[{'environment':{'python':'profile'}}])
                (output/'task_results.jsonl').write_text(json.dumps({'task_id':'rich_1',
                    'status':'SUCCESS','resolved':True,'test_exit_code':0,'agent_patch':'diff patch'})+'\n')
            def execute(code,namespace):namespace['main']=evaluate
            argv=['runner','--base',str(base),'--repo',str(repo),'--output-dir',str(output),
                '--run-dir',str(repo/'runs/pilot'),'--task-id','rich_1','--endpoint',
                'http://127.0.0.1:8120/v1','--gate',str(gate),'--mode','agent']
            with patch.dict(sys.modules,{'calibrate_all_tasks':calibration,
                'gemmacodeagent_full_adapter':adapter,'yaml':yaml}), patch.dict(os.environ,{'G4_MM_ARM':'gemma4_kaggle'}), \
                patch.object(sys,'argv',argv),patch.object(task,'exec',execute,create=True):
                task.main()
            result=json.loads((output/'result.json').read_text())
            context=json.loads((output/'invocation.json').read_text())
            self.assertEqual(result['arm'],'gemma4_kaggle')
            self.assertTrue(result['resolved'])
            self.assertEqual(result['environment_probes'],[{'python':'profile'}])
            self.assertEqual(context['runtime'],str((repo/'envs/eval').resolve()))
            self.assertEqual((output/'agent.patch').read_text(),'diff patch')
            inputs=json.loads((output/'fingerprint.json').read_text())['inputs']
            self.assertEqual(inputs['protocol']['minutes'],15)
            self.assertEqual(inputs['profile'],profile)

if __name__=='__main__':unittest.main()
