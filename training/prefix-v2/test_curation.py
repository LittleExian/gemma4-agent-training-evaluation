"""CPU-only regression checks for conservative curation and feedback matching."""
import copy, gzip, json, tempfile, unittest
from pathlib import Path
from curate import inspect, request_kind, feedback_index, curate, SUMMARY_PREFIX

def example():
    return {'sample_id':'task/attempt/0','task_id':'task','messages':[{'role':'user','content':'fix'}],
            'tools':[{'type':'function'}], 'finish_reason':'tool_calls',
            'target':{'role':'assistant','tool_calls':[{'id':'c','function':{'name':'edit_file',
            'arguments':json.dumps({'old_string':'before','new_string':'after'})}}]}}

class CurationTests(unittest.TestCase):
    def test_noop(self):
        r=example();r['target']['tool_calls'][0]['function']['arguments']='{"old_string":"x","new_string":"x"}'
        self.assertEqual(inspect(r,{})[0],['identical_edit'])
    def test_failed_edit_and_budget(self):
        for error in ['FileEditError','BudgetExceeded']:
            feedback={('task/attempt','c'):json.dumps({'status':'error','error_type':error})}
            self.assertEqual(inspect(example(),feedback)[0],[error])
    def test_failed_shell_is_not_bad_target(self):
        r=example();r['target']['tool_calls'][0]['function']['name']='run_command'
        feedback={('task/attempt','c'):json.dumps({'status':'error','error_type':'CommandError'})}
        self.assertEqual(inspect(r,feedback)[0],[])
    def test_attempt_scope(self):
        feedback={('another/attempt','c'):json.dumps({'status':'error','error_type':'FileEditError'})}
        reasons,obs=inspect(example(),feedback)
        self.assertEqual(reasons,[]);self.assertFalse(obs[0]['matched'])
    def test_summary_classification_uses_prompt_not_response(self):
        r=example();r['target']={'role':'assistant','content':'Conversation Summary'}
        self.assertEqual(request_kind(r),'agent')
        r['tools']=[];r['messages']=[{'role':'user','content':SUMMARY_PREFIX+' here'}]
        self.assertEqual(request_kind(r),'compaction')
    def test_conflicting_feedback_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'requests.jsonl.gz'
            with gzip.open(p,'wt') as f:
                for value in ['one','two']:
                    f.write(json.dumps({'attempt_id':'a','messages':[{'role':'tool','tool_call_id':'c','content':value}]})+'\n')
            with self.assertRaises(ValueError):feedback_index(p)
    def test_history_preserved_and_split_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir()
            r=example();r['messages'].append({'role':'tool','tool_call_id':'older','content':'earlier mistake'})
            (source/'train.jsonl').write_text(json.dumps(r)+'\n');(source/'validation.jsonl').write_text('')
            with gzip.open(source/'requests.jsonl.gz','wt') as f:f.write('')
            attempt={'attempt_id':'task/attempt','task_id':'task','training_split':'train',
                     'environment_valid':True,'selected_success':True,'test_verified_success':True,'gold_passed':True,'empty_failed':True}
            with gzip.open(source/'attempts.jsonl.gz','wt') as f:f.write(json.dumps(attempt)+'\n')
            report=curate(source,root/'out')
            actual=json.loads((root/'out/train.jsonl').read_text())
            self.assertEqual(actual['messages'],r['messages']);self.assertEqual(actual['target'],r['target'])
            self.assertEqual(report['splits']['train']['counts']['kept_rows'],1)
            with self.assertRaises(FileExistsError):curate(source,root/'out')
            (source/'validation.jsonl').write_text(json.dumps(r)+'\n')
            with self.assertRaisesRegex(ValueError,'leakage'):curate(source,root/'out2')

if __name__=='__main__':unittest.main()
