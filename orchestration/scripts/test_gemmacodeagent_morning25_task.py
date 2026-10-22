import importlib.util,pathlib,sys,unittest
from unittest.mock import patch,Mock
import types, tempfile, shutil

class AdaptationTests(unittest.TestCase):
 def setUp(self):
  self.legacy=types.ModuleType('gemmacodeagent_mm_task')
  self.legacy.adapt_source=lambda source,context:source
  self.legacy.main=lambda:None
  self.patch=patch.dict(sys.modules,{'gemmacodeagent_mm_task':self.legacy});self.patch.start();self.addCleanup(self.patch.stop)
  spec=importlib.util.spec_from_file_location('morning',pathlib.Path(__file__).with_name('gemmacodeagent_morning25_task.py'))
  self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
 def fixture(self):
  return '\n'.join(['    raw=yaml.safe_load((submission/"eval_config.yaml").read_text()).get("evaluation",{})','    manifest={"mode":args.mode,"ids":[]}', '                row=result.model_dump(mode="json",exclude={"trace"})',*["x=LiteLlm(extra_body={'thinking_token_budget': 6000, 'chat_template_kwargs': {'enable_thinking': True}})" for _ in range(2)]])
 def test_effective_submission_and_manifest(self):
  value=self.module.adapt_source(self.fixture(),{})
  self.assertIn('materialize_native_submission(submission,out/"effective_submission",raw)',value)
  self.assertIn('"budget_contract":budget_contract',value)
  self.assertIn('## Budget: 56 tool calls, 15 minutes',value)
  self.assertEqual(value.count("'preserve_thinking': False"),2)
  self.assertEqual(value.count("'top_k': 64"),2)
 def test_effective_adapter_stays_inside_submission(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);source=root/'source';dest=root/'effective'
   (source/'adapters/model').mkdir(parents=True)
   (source/'adapters/model/adapter_config.json').write_text('{}')
   (source/'adapters/model/adapter_model.safetensors').write_bytes(b'weights')
   dest.mkdir();(dest/'adapters').symlink_to(source/'adapters',target_is_directory=True)
   (source/'agent.yaml').write_text('name: test')
   (dest/'agent.yaml').symlink_to(source/'agent.yaml')
   module=types.ModuleType('budget_contract')
   module.materialize_submission=lambda *args:(dest,{'version':'test'})
   with patch.dict(sys.modules,{'budget_contract':module}):
    output,metadata=self.module.materialize_native_submission(source,dest,{})
   self.assertFalse((output/'adapters').is_symlink())
   self.assertFalse((output/'agent.yaml').is_symlink())
   for path in (output/'adapters').rglob('*'):
    self.assertTrue(path.resolve().is_relative_to(output.resolve()))
   self.assertEqual((output/'adapters/model/adapter_model.safetensors').read_bytes(),b'weights')
   self.assertEqual((source/'adapters/model/adapter_model.safetensors').read_bytes(),b'weights')
 def test_missing_raw_or_manifest_fails_closed(self):
  for source in [self.fixture().replace('    raw=','    other='),self.fixture().replace('manifest={','different={')]:
   with self.assertRaises(ValueError):self.module.adapt_source(source,{})
 def test_main_installs_only_scoped_hook(self):
  self.legacy.main=Mock()
  self.module.main()
  self.legacy.main.assert_called_once_with()
  self.assertIs(self.legacy.adapt_source,self.module.adapt_source)
 def test_sandbox_setup_failure_is_not_a_model_failure(self):
  with self.assertRaises(RuntimeError):self.module.assert_agent_execution({'error_message':'Sandbox execution error: bad submission'})
  self.module.assert_agent_execution({'resolved':False,'test_exit_code':1,'error_message':'test failure'})
 def test_wrong_request_anchor_count_rejected(self):
  with self.assertRaises(ValueError):self.module.adapt_source(self.fixture().rsplit('\n',1)[0],{})

if __name__=='__main__':unittest.main()
