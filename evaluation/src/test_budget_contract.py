import json,tempfile,unittest
from pathlib import Path
import yaml
from budget_contract import render_budget,materialize_submission
R=Path(__file__).resolve().parents[1]
class BudgetTests(unittest.TestCase):
 def test_all_shipped_submissions(self):
  configs=list((R/'submissions').glob('*/eval_config.yaml'))+[R/'experiments/distributed_model_comparison/submission/eval_config.yaml']
  for cfg in configs:
   raw=yaml.safe_load(cfg.read_text())['evaluation'];text=(cfg.parent/'prompts/system.md').read_text()
   self.assertEqual(render_budget(text,raw),text,str(cfg))
 def test_fractional_budget_and_copy(self):
  source=R/'submissions/v5_6000';original=(source/'prompts/system.md').read_bytes()
  with tempfile.TemporaryDirectory() as d:
   dest,contract=materialize_submission(source,Path(d)/'effective',{'max_tool_calls':60,'max_time_minutes':12.5})
   text=(dest/'prompts/system.md').read_text();self.assertIn('60 tool calls, 12.5 minutes',text)
   self.assertEqual((source/'prompts/system.md').read_bytes(),original)
   self.assertTrue((dest/'agent.yaml').exists());self.assertEqual(contract['max_tool_calls'],60)
   materialize_submission(source,dest,{'max_tool_calls':60,'max_time_minutes':12.5})
   with self.assertRaises(ValueError):materialize_submission(source,dest,{'max_tool_calls':56,'max_time_minutes':15})
 def test_unknown_and_invalid_rejected(self):
  with self.assertRaises(ValueError):render_budget('No budget here',{})
  text=(R/'submissions/v5_6000/prompts/system.md').read_text()
  for raw in [{'max_tool_calls':0},{'max_time_minutes':float('nan')},{'max_tool_calls':True}]:
   with self.assertRaises(ValueError):render_budget(text,raw)
if __name__=='__main__':unittest.main()
