"""No API key, GPU, or online account required. Includes a real offline SDK run."""
import argparse,json,os,pathlib,tempfile,types,unittest
from unittest.mock import patch
from tracking import Tracker,add_arguments,progress_metrics

class FakeRun:
 def __init__(self,kw):
  self.id=kw['id'];self.entity=kw['entity'] or 'tester';self.project=kw['project'];self.url='https://example.invalid/run';self.summary={};self.logs=[];self.finished=None
 def define_metric(self,*a,**kw):pass
 def log(self,m):self.logs.append(m)
 def finish(self,exit_code):self.finished=exit_code
class FakeSDK:
 def __init__(self):self.calls=[]
 def Settings(self,**kw):return kw
 def init(self,**kw):self.calls.append(kw);return FakeRun(kw)
def args(root,**kw):
 d=dict(output=str(root),resume=None,wandb_mode='online',wandb_project='unit-project',wandb_entity=None,wandb_name='unit',wandb_id=None,wandb_new_run=False);d.update(kw);return types.SimpleNamespace(**d)

class TrackingTests(unittest.TestCase):
 def test_disabled_and_env(self):
  with tempfile.TemporaryDirectory() as root:
   sdk=FakeSDK();t=Tracker(args(root,wandb_mode='disabled'),{},0,sdk);self.assertFalse(sdk.calls);t.finish()
  with patch.dict(os.environ,{},clear=True):
   p=argparse.ArgumentParser();add_arguments(p);self.assertEqual(p.parse_args([]).wandb_mode,'disabled')
  with patch.dict(os.environ,{'WANDB_PROJECT':'my-project','WANDB_ENTITY':'my-team'},clear=True):
   p=argparse.ArgumentParser();add_arguments(p);a=p.parse_args([]);self.assertEqual((a.wandb_mode,a.wandb_project,a.wandb_entity),('online','my-project','my-team'))
 def test_resume_and_config_whitelist(self):
  with tempfile.TemporaryDirectory() as root:
   sdk=FakeSDK();t=Tracker(args(root),{'lr':2e-5,'api_key':'not-for-upload','data':'private-path'},0,sdk)
   self.assertNotIn('api_key',sdk.calls[0]['config']);self.assertNotIn('data',sdk.calls[0]['config']);rid=t.run.id
   ck=pathlib.Path(root)/'checkpoint-3';ck.mkdir();t.checkpoint(ck,3,.1);t.finish()
   restored=Tracker(args(root,resume=str(ck)),{'lr':2e-5},3,sdk);self.assertEqual(restored.run.id,rid);self.assertEqual(sdk.calls[-1]['resume'],'must');restored.finish()
   with self.assertRaises(ValueError):Tracker(args(root,resume=str(ck),wandb_entity='different-team'),{},3,sdk)
   fork=Tracker(args(root,resume=str(ck),wandb_entity='different-team',wandb_new_run=True),{},3,sdk);self.assertNotEqual(fork.run.id,rid);self.assertEqual(sdk.calls[-1]['resume'],'never');fork.finish()
 def test_metrics_aggregate_all_ranks(self):
  m=progress_metrics(4,10,2,1,20,2.,8.,4,[[100,20,5,15,60,61,1.5,100],[300,40,10,30,70,72,1.8,300]])
  self.assertEqual(m['train/sequence_tokens_per_second'],200.);self.assertEqual(m['train/eta_seconds'],12.);self.assertEqual(m['train/progress_percent'],40.);self.assertEqual(m['gpu/rank_1/peak_allocated_gib'],70)
 def test_real_offline_sdk(self):
  with tempfile.TemporaryDirectory() as root:
   t=Tracker(args(root,wandb_mode='offline',wandb_entity=None),{'lr':2e-5,'world_size':2},0)
   run=t.run;t.log({'train/optimizer_step':1,'train/loss':1.25});t.finish();self.assertIsNone(t.run)
   state=json.loads((pathlib.Path(root)/'wandb_run.json').read_text());self.assertEqual(state['mode'],'offline')
   self.assertTrue(list(pathlib.Path(root).glob('wandb/offline-run-*/run-*.wandb')))
   data=next(pathlib.Path(root).glob('wandb/offline-run-*/run-*.wandb')).read_bytes()
   for field in [b'train/loss',b'1.25',b'training_exit_code']:
    self.assertIn(field,data)
if __name__=='__main__':unittest.main()
