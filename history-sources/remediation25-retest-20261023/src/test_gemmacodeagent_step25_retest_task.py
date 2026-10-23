import importlib.util,pathlib,sys,types,unittest
from unittest.mock import patch,Mock

class RequestTests(unittest.TestCase):
 def setUp(self):
  self.original=types.ModuleType('gemmacodeagent_morning25_task')
  self.original.adapt_source=lambda source,context:source
  self.legacy=types.ModuleType('gemmacodeagent_mm_task');self.legacy.main=lambda:None
  self.patcher=patch.dict(sys.modules,{'gemmacodeagent_morning25_task':self.original,'gemmacodeagent_mm_task':self.legacy});self.patcher.start();self.addCleanup(self.patcher.stop)
  spec=importlib.util.spec_from_file_location('new_task',pathlib.Path(__file__).with_name('gemmacodeagent_step25_retest_task.py'));self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
 def fixture(self):return '\n'.join(["x=LiteLlm(extra_body={'thinking_token_budget': 6000, 'top_k': 64, 'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': False}})" for _ in range(2)])
 def test_both_request_routes_use_explicit_seed_and_topk(self):
  result=self.module.adapt_source(self.fixture(),{})
  self.assertEqual(result.count('seed=42,'),2);self.assertEqual(result.count("'top_k': -1"),2);self.assertNotIn("'top_k': 64",result)
  self.assertEqual(result.count("'preserve_thinking': False"),2)
 def test_changed_or_duplicate_request_anchors_fail_closed(self):
  for source in [self.fixture().splitlines()[0],self.fixture()+'\n'+self.fixture(),self.fixture().replace("'top_k': 64","'top_k': 12")]:
   with self.assertRaises(ValueError):self.module.adapt_source(source,{})

 def test_main_installs_seeded_adaptation(self):
  self.legacy.main=Mock()
  self.module.main()
  self.legacy.main.assert_called_once_with()
  self.assertIs(self.legacy.adapt_source,self.module.adapt_source)

if __name__=='__main__':unittest.main()
