import unittest
import json
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import compare


def row(task, number, passed, excluded=False, timeout=False, patch='same'):
    return dict(instance_id=task, round=number, repo=task.rsplit('_', 1)[0], resolved=passed,
                status='preflight_excluded' if excluded else 'agent_timeout' if timeout else 'completed',
                agent_timed_out=timeout, agent_seconds=None if excluded else 10.0,
                full_seconds=None if excluded else 12.0, tool_calls=1, llm_calls=2,
                patch_sha256=None if excluded else patch, patch_bytes=None if excluded else 0,
                source_result_sha256='hash', verification_exit_codes=[] if excluded else [0 if passed else 1])

class CompareTests(unittest.TestCase):
    def test_teich_identity_and_context_failure_are_separate_from_timeout(self):
        source=self.fixture()
        source[0]={**source[0], 'agent_timed_out':False, 'status':'completed', 'agent_error_code':'agent_context_limit'}
        report,tasks=compare.analyze(source)
        self.assertEqual(report['method'],'teich-best')
        self.assertEqual(report['rounds'][0]['context_limit_ids'],['repo_0'])
        self.assertEqual(report['rounds'][0]['context_limit_count'],1)
        self.assertEqual(report['rounds'][0]['timeouts'],0)
        self.assertEqual(compare.flatten(tasks[1])['r1_agent_error_code'],'agent_context_limit')
    def fixture(self):
        return [row('repo_'+str(i), n, pattern[n-1]=='1', timeout=(i==0), patch=str(n) if i==1 else 'same')
                for i,pattern in enumerate(('111','010','000')) for n in (1,2,3)] + [row('excluded_1', n, False, True) for n in (1,2,3)]
    def test_patterns_exclude_unscored_and_timeout_can_pass(self):
        report, tasks=compare.analyze(self.fixture())
        self.assertEqual(report['eligible_tasks'],3)
        self.assertEqual(report['excluded_tasks'],['excluded_1'])
        self.assertEqual(report['patterns']['111']['count'],1)
        self.assertEqual(report['patterns']['010']['count'],1)
        self.assertEqual(report['patterns']['000']['count'],1)
        self.assertEqual(report['all_passed_count'],1)
        self.assertEqual(report['all_failed_count'],1)
        self.assertEqual(report['variable_correctness_count'],1)
        self.assertEqual(report['rounds'][0]['timeouts_that_passed'],1)
        self.assertEqual(next(t for t in tasks if t['instance_id']=='excluded_1')['pattern'],'not_scored')
    def test_pairwise_direction_and_patch_variation(self):
        report,tasks=compare.analyze(self.fixture())
        pair=report['pairwise'][0]
        self.assertEqual(pair['from_round'],1)
        self.assertEqual(pair['to_round'],2)
        self.assertEqual(pair['newly_passed_ids'],['repo_1'])
        self.assertEqual(pair['lost_pass_ids'],[])
        self.assertEqual(pair['pass_jaccard'],0.5)
        self.assertFalse(next(t for t in tasks if t['instance_id']=='repo_1')['patches_all_equal'])
        self.assertEqual(report['patches_identical_all_rounds'],2)
        self.assertEqual(report['identical_empty_patch_ids'],['repo_0','repo_2'])
        self.assertEqual(report['identical_nonempty_patch_ids'],[])
    def test_same_patch_different_grade_is_explicitly_reported(self):
        source=self.fixture()
        source=[{**r,'patch_sha256':'shared','patch_bytes':7} if r['instance_id']=='repo_1' else r for r in source]
        report,_=compare.analyze(source)
        self.assertEqual(report['pairwise'][0]['same_patch_different_correctness_ids'],['repo_1'])
        note=compare.same_patch_note(report)
        self.assertIn('同一原始补丁却判定不同',note)
        self.assertIn('repo_1',note)
        self.assertNotIn('未发现',note)
    def test_missing_or_duplicate_round_fails(self):
        source=self.fixture()
        with self.assertRaises(ValueError):compare.analyze(source[:-1])
        with self.assertRaises(ValueError):compare.analyze(source+[source[0]])
    def test_exclusion_status_drift_fails(self):
        source=self.fixture(); source[-1]={**source[-1], 'status':'completed'}
        with self.assertRaises(ValueError):compare.analyze(source)
    def test_known_context_and_status_not_silently_scored(self):
        source=self.fixture();source[0]={**source[0], 'status':'infra_error'}
        with self.assertRaisesRegex(ValueError,'Non-evaluation'):compare.analyze(source)
    def test_full_export_counts_and_explicit_exclusion(self):
        source=[row('task_'+str(i),n,i<50) for i in range(126) for n in (1,2,3)]
        source += [row('excluded_'+str(i),n,False,True) for i in range(3) for n in (1,2,3)]
        with TemporaryDirectory() as td:
            root=Path(td);(root/'source-scalars.json').write_text(json.dumps(source))
            with patch.object(compare,'__file__',str(root/'compare.py')):compare.main()
            report=json.loads((root/'output/comparison.json').read_text())
            self.assertEqual([x['resolved'] for x in report['rounds']],[50,50,50])
            with (root/'output/task-results-long.csv').open() as handle:long=list(csv.DictReader(handle))
            self.assertEqual(len(long),387)
            self.assertEqual(sum(x['correctness']=='not_scored' for x in long),9)
            with (root/'output/task-comparison.csv').open() as handle:wide=list(csv.DictReader(handle))
            self.assertEqual(len(wide),129)
            with (root/'output/changed-tasks.csv').open() as handle:self.assertEqual(len(list(csv.DictReader(handle))),0)
            self.assertIn('GPU2、GPU3',(root/'output/COMPARISON.md').read_text())
    def test_main_rejects_partial_collection(self):
        with TemporaryDirectory() as td:
            root=Path(td);(root/'source-scalars.json').write_text(json.dumps(self.fixture()))
            with patch.object(compare,'__file__',str(root/'compare.py')):
                with self.assertRaisesRegex(ValueError,'complete129'):compare.main()

if __name__=='__main__':unittest.main()
