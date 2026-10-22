import unittest
from assess_results import assess

def row(arm,task,solved,seconds=10):
 return {'arm':arm,'task_id':task,'resolved':solved,'elapsed_seconds':seconds,'process_exit_code':0,'status':'SUCCESS'}
class GateTests(unittest.TestCase):
 def test_pipeline_success_is_not_solve(self):
  r=assess([row('base','a',True),row('lora','a',False)],['a'])
  self.assertFalse(r['passed']);self.assertEqual(r['baseline_only'],['a'])
 def test_missing_not_dropped(self):
  with self.assertRaises(ValueError):assess([row('base','a',True)],['a'])
 def test_duplicate_rejected(self):
  with self.assertRaises(ValueError):assess([row('base','a',True)]*2,['a'])
 def test_slow_candidate_fails(self):
  self.assertFalse(assess([row('base','a',True),row('lora','a',True,20)],['a'])['passed'])
 def test_infra_failure_rejected(self):
  r=row('lora','a',False);r['process_exit_code']=1
  with self.assertRaises(ValueError):assess([row('base','a',True),r],['a'])
 def test_equal_passes_development_gate(self):
  self.assertTrue(assess([row('base','a',True),row('lora','a',True)],['a'])['passed'])
if __name__=='__main__':unittest.main()
