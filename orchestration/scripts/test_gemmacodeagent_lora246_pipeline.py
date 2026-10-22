import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gemmacodeagent_lora246_pipeline as pipeline


class LoRA246PipelineTests(unittest.TestCase):
    def test_scope_is_fixed_and_full_is_fresh(self):
        ids = ['t'+str(i) for i in range(129)]
        pipeline.validate_scope(ids[:16], ids)
        for pilot, full in [(ids[:15], ids), (ids[:16], ids[:-1]), (ids[:16], ids[:128]+[ids[0]])]:
            with self.assertRaises(ValueError):
                pipeline.validate_scope(pilot, full)

    def test_final_gate_binds_adapter_not_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root/'deployment_gate.json'
            gate = {'passed': True, 'zero_adapter_max_logit_delta': 0.,
                    'trained_adapter_max_logit_delta': .5, 'export': {'rank': 16, 'tensor_count': 820}}
            path.write_text(json.dumps(gate))
            manifest = {'checkpoint': 246, 'adapter': {'weights_sha256': 'final'},
                        'deployment_gate': {'path': str(path), 'sha256': pipeline.digest(path),
                                            'adapter_weights_sha256': 'final'}}
            pipeline.validate_deployment_gate(path, manifest)
            for altered in [{**manifest, 'checkpoint': 1},
                            {**manifest, 'adapter': {'weights_sha256': 'different'}}]:
                with self.assertRaises(ValueError):
                    pipeline.validate_deployment_gate(path, altered)
            path.write_text(json.dumps({**gate, 'passed': False}))
            with self.assertRaises(ValueError):
                pipeline.validate_deployment_gate(path, manifest)

    def test_packaged_copy_discovery_and_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            adapter = root/'submissions/mm15_6000/adapters'/pipeline.ARM
            adapter.mkdir(parents=True)
            source = root/'checkpoint-246/adapter'
            source.mkdir(parents=True)
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                (adapter/name).write_text(name)
                (source/name).write_text(name)
            manifest = {'checkpoint': 246, 'base_model': {'path': str(root/'models'/pipeline.BASE_ALIAS), 'weights_sha256': pipeline.BASE_SHA},
                'adapter': {'name': pipeline.ARM, 'path': str(adapter), 'source_path': str(source),
                    'weights_sha256': pipeline.digest(adapter/'adapter_model.safetensors'),
                    'config_sha256': pipeline.digest(adapter/'adapter_config.json')},
                'discovery': {'adapters': [{'name': pipeline.ARM, 'path': str(adapter/'adapter_model.safetensors')}]}}
            self.assertEqual(pipeline.validate_package(root, manifest), adapter)
            (adapter/'adapter_config.json').write_text('changed')
            with self.assertRaises(ValueError):
                pipeline.validate_package(root, manifest)

    def test_real_official_discovery_manifest_uses_weight_file_path(self):
        path = Path(__file__).resolve().parent.parent/'artifacts/lora246-eval-20261021/adapter-package.json'
        if not path.exists():
            self.skipTest('Captured official discovery report is not in this checkout')
        manifest = json.loads(path.read_text())
        root = Path(manifest['base_model']['path']).parent.parent
        def digest(path):
            key = 'config_sha256' if path.name == 'adapter_config.json' else 'weights_sha256'
            return manifest['adapter'][key]
        with patch.object(pipeline, 'streaming_digest', side_effect=digest), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'is_symlink', return_value=False):
            pipeline.validate_package(root, manifest)
            changed = {**manifest, 'discovery': {'adapters': [{'name': pipeline.ARM,
                       'path': manifest['adapter']['path'], 'rank': 16}]}}
            with self.assertRaises(ValueError):
                pipeline.validate_package(root, changed)

    def test_health_requires_actual_adapter_fourway_tools_and_budget(self):
        valid = {'passed': True, 'concurrency': 4, 'models': [pipeline.ARM],
                 'thinking_tokens': {pipeline.ARM: 16}, 'rows': [{'request': {'model': pipeline.ARM, 'tools': [{}]}}]}
        pipeline.validate_health(valid)
        for changes in ({'passed': False}, {'concurrency': 1}, {'models': ['base']},
                        {'thinking_tokens': {}}, {'rows': []}):
            with self.assertRaises(RuntimeError):
                pipeline.validate_health({**valid, **changes})

    def test_exceptions_are_never_scored_resolved(self):
        rows = [{'instance_id': 'x', 'status': 'preflight_excluded', 'resolved': True}]
        with self.assertRaises(RuntimeError):
            pipeline.summarize(rows, ['x'], 'full', ['x'])
        rows[0]['resolved'] = False
        report = pipeline.summarize(rows, ['x'], 'full', ['x'])
        self.assertEqual(report['excluded_ids'], ['x'])
        self.assertEqual(report['eligible'], 0)
        self.assertEqual(report['resolved'], 0)

    def test_launch_partial_failure_still_cleans_owned_tree(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def checked(command, *args, **kwargs):
                if 'launch' in command:
                    state = Path(command[command.index('--state')+1])
                    pipeline.atomic_json(state/'process.json', {'pid': 1})
                    raise RuntimeError('Launch failed')
            with patch.object(pipeline, 'checked_process', side_effect=checked), \
                 patch.object(pipeline, 'cleanup_secondary') as cleanup:
                with self.assertRaisesRegex(RuntimeError, 'Launch failed'):
                    pipeline.start_models(root, root/'adapter', root/'services', {})
            self.assertEqual(cleanup.call_args.args[0], [{'pid': 1}])

    def test_two_cards_launch_health_and_exact_eight_slots(self):
        calls = []
        valid = {'passed': True, 'concurrency': 4, 'models': [pipeline.ARM],
                 'thinking_tokens': {pipeline.ARM: 16}, 'rows': [{'request': {'model': pipeline.ARM, 'tools': [{}]}}]}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def checked(command, *args, **kwargs):
                calls.append(command)
                state = Path(command[command.index('--state')+1])
                if 'launch' in command:
                    gpu = int(command[command.index('--gpu')+1])
                    pipeline.atomic_json(state/'process.json', {'pid': gpu+1})
                else:
                    pipeline.atomic_json(state/'health.json', valid)
            with patch.object(pipeline, 'checked_process', side_effect=checked):
                models, _ = pipeline.start_models(root, root/'adapter', root/'services', {})
        launches = [command for command in calls if 'launch' in command]
        self.assertEqual([command[command.index('--gpu')+1] for command in launches], ['0', '2'])
        self.assertEqual([command[command.index('--port')+1] for command in launches], ['8140', '8142'])
        self.assertTrue(all(command[command.index('--slots')+1] == '4' for command in launches))
        self.assertEqual(models, [{'pid': 1}, {'pid': 3}])

    def test_runner_dispatch_routes_only_two_endpoints_four_workers_each(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); gate = root/'gate.json'
            gate.write_text('{"quarantined_ids":[],"nondiscriminating_ids":[]}')
            records = [{'instance_id': 'a', 'status': 'completed', 'resolved': False,
                        'validations': [{'infra_error': False}]}]
            with patch.object(pipeline, 'checked_process') as checked, \
                 patch.object(pipeline, 'records_at', return_value=records):
                report = pipeline.evaluate(root, root/'base', gate, pipeline.ARM, 'pilot16', ['a'],
                    root/'runs/test', [{'pid': 42}], {}, root/'fingerprint.json')
            command = checked.call_args.args[0]
            endpoints = command[command.index('--endpoints')+1:command.index('--workers')]
            workers = command[command.index('--workers')+1:command.index('--fingerprint')]
            self.assertEqual(endpoints, ['http://127.0.0.1:8140/v1', 'http://127.0.0.1:8142/v1'])
            self.assertEqual(workers, ['4', '4'])
            self.assertEqual(checked.call_args.args[2]['G4_MM_ARM'], pipeline.ARM)
            self.assertEqual(report['eligible'], 1)

    def test_protocol_binds_source_adapter_and_original_weight(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); experiment = root/'experiments/distributed_model_comparison'
            for relative in ['src/controller.py', 'submissions/mm15_6000/agent.yaml',
                             'environment/installed_code_hashes.json', 'base/data/tasks.jsonl',
                             'gate.json', 'package.json', 'deployment.json',
                             'models/'+pipeline.BASE_ALIAS+'/config.json',
                             'experiments/distributed_model_comparison/pilot16.json']:
                path = root/relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(relative)
            def streaming(path):
                return pipeline.BASE_SHA if path.name == 'model.safetensors' else pipeline.digest(path)
            with patch.object(pipeline, 'streaming_digest', side_effect=streaming), \
                 patch.object(pipeline.subprocess, 'check_output', return_value='frozen'):
                protocol = pipeline.build_protocol(root, root/'base', root/'gate.json', experiment,
                                                   root/'package.json', root/'deployment.json')
            self.assertEqual(protocol['gpus'], [0, 2])
            self.assertEqual(protocol['total_slots'], 8)
            self.assertEqual(protocol['protocol']['top_k'], 64)
            self.assertIn('config.json', protocol['base_metadata_sha256'])
            self.assertIn('submissions/mm15_6000/agent.yaml', protocol['source_sha256'])
            with patch.object(pipeline, 'streaming_digest', return_value='different'):
                with self.assertRaises(ValueError):
                    pipeline.build_protocol(root, root/'base', root/'gate.json', experiment,
                                            root/'package.json', root/'deployment.json')

    def test_runtime_check_only_validates_without_installing(self):
        with patch.object(pipeline.subprocess, 'run') as run:
            pipeline.verify_runtime(Path('/isolated'))
        self.assertEqual(run.call_count, 2)
        self.assertIn('check', run.call_args_list[0].args[0])
        self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))

    def run_mock(self, root, failed=False):
        experiment = root/'experiments/distributed_model_comparison'
        experiment.mkdir(parents=True)
        ids = ['t'+str(i) for i in range(129)]
        for filename, values in [('pilot16.json', ids[:16]), ('full129.json', ids)]:
            (experiment/filename).write_text(json.dumps({'tasks': values}))
        gate = root/'gate.json'; gate.write_text(json.dumps({'quarantined_ids': ['t128'], 'nondiscriminating_ids': ['t126', 't127']}))
        manifest = root/'package.json'; manifest.write_text(json.dumps({'adapter': {'source_path': '/checkpoint-246/adapter'}}))
        deployment = root/'deployment.json'; deployment.write_text('{}')
        args = SimpleNamespace(root=root, base=root/'base', gate=gate, run_dir=root/'runs/test',
                               manifest=manifest, deployment_gate=deployment)
        calls = []
        def evaluate(*values):
            scope = values[4]
            calls.append(scope)
            if failed and scope == 'pilot16':
                raise RuntimeError('pilot infrastructure failure')
            return {'passed': True, 'scope': scope}
        with patch.object(pipeline, 'validate_package', return_value=root/'adapter'), \
             patch.object(pipeline, 'validate_deployment_gate'), \
             patch.object(pipeline, 'build_protocol', return_value={'fixed': True}), \
             patch.object(pipeline, 'verify_runtime'), \
             patch.object(pipeline, 'start_models', return_value=([{'pid': 1}], 1.)), \
             patch.object(pipeline, 'evaluate', side_effect=evaluate), \
             patch.object(pipeline, 'cleanup_secondary', return_value=[]) as cleanup:
            if failed:
                with self.assertRaisesRegex(RuntimeError, 'pilot infrastructure failure'):
                    pipeline.run(args)
            else:
                pipeline.run(args)
        return calls, json.loads((args.run_dir/'pipeline-status.json').read_text()), cleanup

    def test_complete_lifecycle_pilot_then_fresh_full_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            calls, state, cleanup = self.run_mock(Path(td))
        self.assertEqual(calls, ['pilot16', 'full'])
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(cleanup.call_count, 2)

    def test_failed_pilot_blocks_full_preserving_error_and_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            calls, state, cleanup = self.run_mock(Path(td), failed=True)
        self.assertEqual(calls, ['pilot16'])
        self.assertEqual(state['status'], 'blocked')
        self.assertIn('pilot infrastructure failure', state['error'])
        cleanup.assert_called_once()


if __name__ == '__main__':
    unittest.main()
