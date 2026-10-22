import json
import os
from unittest.mock import Mock, patch
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import remediation_training_supervisor as s


class SupervisorTests(unittest.TestCase):
    def args(self, root):
        return SimpleNamespace(root=root, source=root/'src', data=root/'train.jsonl',
            output=root/'out', hf_home=root/'cache', python=Path('/usr/bin/python3'),
            credential_file=root/'key', scope='full', gpus='2,3', cap_gib=84,
            wandb_entity='entity', run_name='remediation', manifest=None)

    def test_fixed_two_gpu_recipe(self):
        args=self.args(Path('/tmp/root'))
        command=s.build_command(args, 'entity')
        self.assertIn('--nproc_per_node=2', command)
        self.assertEqual(command[command.index('--objective')+1], 'task_mean')
        self.assertEqual(command[command.index('--accum')+1], '4')
        self.assertEqual(command[command.index('--steps')+1], '0')
        self.assertEqual(command[command.index('--save-every')+1], '5')
        self.assertEqual(s.TOTAL_STEPS['full'], 356)

    def test_gpu_foreign_small_process_is_recorded_not_rejected(self):
        inv='2, GPU-two, '+s.GPU_NAME+', 97887, 90000\n3, GPU-three, '+s.GPU_NAME+', 97887, 97000\n'
        self.assertEqual(s.select_gpus(inv, 'GPU-two, 123\n', '2,3'), ['GPU-two','GPU-three'])
        with self.assertRaises(ValueError):
            s.select_gpus(inv.replace('90000','80000'), '', '2,3')
        with self.assertRaises(ValueError):
            s.select_gpus(inv, '', '2,2')

    def test_environment_numeric_ids_and_secrets_not_inherited_resume(self):
        args=self.args(Path('/tmp/root'))
        env=s.training_environment(args, ['GPU-two','GPU-three'], 'secret', 'entity', {'WANDB_RESUME':'allow'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2,3')
        self.assertNotIn('WANDB_RESUME',env)
        self.assertEqual(env['WANDB_API_KEY'],'secret')

    def test_credential_mode_and_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'key';p.write_text('secret\n');p.chmod(0o600)
            self.assertEqual(s.read_credential(p),'secret')
            p.chmod(0o644)
            with self.assertRaises(ValueError):s.read_credential(p)
            q=Path(temp)/'link';q.symlink_to(p)
            with self.assertRaises(OSError):s.read_credential(q)

    def test_rows_fixed_full_and_nonempty_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'data';p.write_text('{}\n'*2841)
            self.assertEqual(s.validate_recipe_data(p,'full'),2841)
            p.write_text('{}\n')
            with self.assertRaises(ValueError):s.validate_recipe_data(p,'full')
            self.assertEqual(s.validate_recipe_data(p,'smoke'),1)

    def test_metrics_strip_samples_and_incomplete_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for rank in range(2):
                (root/f'metrics.rank{rank}.jsonl').write_text('{"step": 1, "sample_ids": ["private"]}\n{"step":')
            self.assertEqual(s.last_metrics(root),{'0':{'step':1},'1':{'step':1}})

    def test_full_completion_two_rank_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cp=root/'checkpoint-356';cp.mkdir()
            for rank in range(2):
                (root/f'metrics.rank{rank}.jsonl').write_text(json.dumps({'step':356,'loss':.2,'gradient_norm':1.,'adapter_delta_max':.1})+'\n')
                (cp/f'rng.rank{rank}.pt').write_bytes(b'x')
            (cp/'COMPLETE.json').write_text(json.dumps({'step':356,'world_size':2}))
            for p in (root/'base_audit.json',cp/'adapter_audit.json'):p.write_text('{"passed":true}')
            self.assertEqual(s.verify_completion(root,'full')['world_size'],2)
            (cp/'COMPLETE.json').write_text('{"step":356,"world_size":3}')
            with self.assertRaises(ValueError):s.verify_completion(root,'full')

    def test_output_scope_and_freshness(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ('source', 'hf'):
                (root/name).mkdir()
            (root/'source/train.py').write_text('')
            (root/'data.jsonl').write_text('{}\n')
            args = SimpleNamespace(root=root, source=root/'source', data=root/'data.jsonl', hf_home=root/'hf',
                output=root/'new', credential_file=Path('/workspace/.credentials/key'), manifest=None, run_name='run', scope='smoke', gpus='2,3', cap_gib=84, python=Path('/usr/bin/python3'), wandb_entity='entity')
            s.validate_paths(args)
            (root/'new').mkdir()
            with self.assertRaises(ValueError):
                s.validate_paths(args)
            with self.assertRaises(ValueError):
                s.validate_paths(SimpleNamespace(**{**vars(args), 'output': Path('/tmp/outside')}))

    def test_completion_requires_all_ranks_and_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            cp = output/'checkpoint-1'
            cp.mkdir()
            for rank in range(2):
                (output/f'metrics.rank{rank}.jsonl').write_text(json.dumps({'step': 1, 'loss': .5,
                    'gradient_norm': .2, 'adapter_delta_max': .01})+'\n')
                (cp/f'rng.rank{rank}.pt').write_bytes(b'rng')
            for path, value in ((cp/'COMPLETE.json', {'step': 1, 'world_size': 2}),
                                (cp/'adapter_audit.json', {'passed': True}),
                                (output/'base_audit.json', {'passed': True}),
                                (output/'base_audit_after.json', {'passed': True}),
                                (output/'SMOKE_PASS.json', {'passed': True, 'steps': 1, 'world_size': 2,
                                    'reload_max_abs_logit_difference': 0})):
                path.write_text(json.dumps(value))
            self.assertTrue(s.verify_completion(output, 'smoke')['passed'])
            (output/'base_audit_after.json').write_text('{"passed":true,"different":true}')
            with self.assertRaises(ValueError):
                s.verify_completion(output, 'smoke')
            (output/'base_audit_after.json').write_text('{"passed":true}')
            (cp/'rng.rank1.pt').unlink()
            with self.assertRaises(ValueError):
                s.verify_completion(output, 'smoke')

    def test_manifest_binds_data_and_exact_python_source_subset(self):
        actual = {'data_sha256': 'data-hash', 'source_sha256': {'train.py': 'source-hash'}}
        expected = {'training_data_sha256': 'data-hash', 'source_sha256':
                    {'train.py': 'source-hash', 'README.md': 'ignored'}}
        s.verify_manifest(actual, expected)
        for changed in ({**expected, 'training_data_sha256': 'other'},
                        {**expected, 'source_sha256': {'train.py': 'other'}},
                        {**expected, 'source_sha256': {'train.py': 'source-hash', 'extra.py': 'other'}}, {}):
            with self.assertRaises(ValueError):
                s.verify_manifest(actual, changed)

    def test_host_memory_gate_requires_24_gib(self):
        self.assertEqual(s.validate_host_memory(24*2**30), 24*2**30)
        with self.assertRaises(ValueError):
            s.validate_host_memory(24*2**30-1)

    def test_bounded_lifecycle_records_outcomes_without_secrets(self):
        for outcome in ('completed', 'child_failed', 'timeout', 'interrupt', 'manifest_drift', 'host_low'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                (root/'source').mkdir(); (root/'hf').mkdir(); (root/'work').mkdir()
                (root/'source/train.py').write_text('# frozen source\n')
                (root/'data.jsonl').write_text('{}\n')
                (root/'manifest.json').write_text(json.dumps({'training_data_sha256': s.fingerprint(root/'data.jsonl'),
                    'source_sha256': {'train.py': s.fingerprint(root/'source/train.py')}}))
                if outcome == 'manifest_drift':
                    (root/'source/train.py').write_text('# unapproved change\n')
                (root/'work/wandb-connectivity.json').write_text('{"passed":true,"entity":"tested-account"}')
                args = SimpleNamespace(root=root, source=root/'source', data=root/'data.jsonl', hf_home=root/'hf',
                    output=root/'output', credential_file=Path('/workspace/.credentials/key'),
                    manifest=root/'manifest.json', run_name='new-run', scope='smoke', gpus='2,3', cap_gib=84, python=Path('/usr/bin/python3'), wandb_entity='entity')
                inventory = '\n'.join(f'{i}, GPU-{i}, {s.GPU_NAME}, 98304, 95000' for i in (2, 3))
                process = Mock(pid=123, returncode=7 if outcome == 'child_failed' else 0)
                process.poll.side_effect = [None, 0]
                fake_psutil = Mock()
                fake_psutil.Process.return_value.create_time.return_value = 123.0
                fake_psutil.virtual_memory.return_value.available = (16 if outcome == 'host_low' else 32)*2**30
                with patch.dict('sys.modules', {'psutil': fake_psutil}), \
                        patch.object(s.subprocess, 'check_output', side_effect=[inventory, '']), \
                        patch.object(s.subprocess, 'Popen', return_value=process) as popen, \
                        patch.object(s, 'read_credential', return_value='do-not-expose-key'), \
                        patch.object(s, 'verify_completion', return_value={'passed': True}), \
                        patch.object(s, 'release_owned_model', return_value={'terminated': [123]}) as cleanup, \
                        patch.object(s.time, 'CLOCK_BOOTTIME', 7, create=True), \
                        patch.object(s.time, 'clock_gettime', return_value=100.0), \
                        patch.object(s.time, 'sleep', side_effect=KeyboardInterrupt if outcome == 'interrupt' else None), \
                        patch.dict(s.TIMEOUT_SECONDS, {'smoke': -1 if outcome == 'timeout' else 7200}):
                    code = s.run(args)
                exit_record = json.loads((args.output/'exit.json').read_text())
                self.assertEqual(code, 0 if outcome == 'completed' else 130 if outcome == 'interrupt' else 1)
                self.assertEqual(exit_record['state'], 'completed' if outcome == 'completed' else 'failed')
                preflight_failed = outcome in ('manifest_drift', 'host_low')
                self.assertEqual(cleanup.call_count, 0 if outcome == 'completed' or preflight_failed else 1)
                if preflight_failed:
                    popen.assert_not_called()
                else:
                    self.assertTrue(popen.call_args.kwargs['start_new_session'])
                    self.assertEqual(popen.call_args.kwargs['env']['WANDB_API_KEY'], 'do-not-expose-key')
                for artifact in args.output.iterdir():
                    self.assertNotIn('do-not-expose-key', artifact.read_text())
                if outcome != 'manifest_drift':
                    self.assertEqual((args.output/'input-manifest.json').read_text(), (root/'manifest.json').read_text())


    def test_worker_caps_allocator_before_running_source(self):
        with tempfile.TemporaryDirectory() as temp:
            train = Path(temp)/'train.py'; train.write_text('')
            torch = Mock()
            torch.cuda.get_device_properties.return_value.total_memory = 96*2**30
            with patch.dict('sys.modules', {'torch': torch}), patch.dict(os.environ, {'LOCAL_RANK':'1'}), \
                    patch('runpy.run_path') as execute, patch('sys.argv', ['old']):
                self.assertEqual(s.worker(['--worker','--cap-gib','84','--train-script',str(train),'--steps','0']),0)
                torch.cuda.set_device.assert_called_once_with(1)
                torch.cuda.set_per_process_memory_fraction.assert_called_once_with(84/96,1)
                execute.assert_called_once_with(str(train),run_name='__main__')
            with self.assertRaises(ValueError):
                s.worker(['--worker','--cap-gib','90','--train-script',str(train)])

    def test_process_snapshot_does_not_mutate_or_kill_foreign_processes(self):
        fake=Mock();fake.Process.return_value.uids.return_value.real=1006
        fake.Process.return_value.pid=123;fake.Process.return_value.create_time.return_value=12.0
        with patch.dict('sys.modules',{'psutil':fake}):
            self.assertEqual(s.process_identities('GPU-two,123\n')[0]['uid'],1006)
            fake.Process.return_value.kill.assert_not_called()
            fake.Process.return_value.terminate.assert_not_called()

    def test_main_worker_dispatch(self):
        with patch('sys.argv',['supervisor','--worker','--cap-gib','84']),patch.object(s,'worker',return_value=0) as worker:
            self.assertEqual(s.main(),0)
            worker.assert_called_once()

if __name__=='__main__':unittest.main()
