import hashlib,importlib.util,json,pathlib,sys,tempfile,types,unittest
from unittest.mock import Mock,patch

class ControllerTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.r=pathlib.Path(self.temp.name);self.svc=types.ModuleType('gemmacodeagent_step25_retest_server');self.svc.cleanup=Mock(return_value={'terminated':[]})
  self.patch=patch.dict(sys.modules,{'gemmacodeagent_step25_retest_server':self.svc});self.patch.start();self.addCleanup(self.patch.stop)
  spec=importlib.util.spec_from_file_location('controller',pathlib.Path(__file__).with_name('gemmacodeagent_step25_retest_controller.py'))
  self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
  self.m.ROOT=self.r;self.m.BASE=self.r/'base';self.m.GATE=self.r/'gate.json';self.m.CHECKPOINT=self.r/'checkpoint-25'
  self.ids=[f'task-{i}' for i in range(129)]
  self.records=[{'instance_id':name,'status':'preflight_excluded' if i<3 else 'completed','resolved':i==4,'validations':[{'infra_error':False}]} for i,name in enumerate(self.ids)]
  self.write('base/data/tasks.jsonl','\n'.join(json.dumps({'instance_id':n}) for n in self.ids))
  self.gate={'dataset_sha256':self.m.digest(self.m.BASE/'data/tasks.jsonl'),'quarantined_ids':self.ids[:1],'nondiscriminating_ids':self.ids[1:3],'task_bindings':{}}
  self.write('gate.json',json.dumps(self.gate));self.write('checkpoint-25/COMPLETE.json',json.dumps({'step':25,'world_size':2}))
  for name in ['adapter_config.json','adapter_model.safetensors']:
   self.write('checkpoint-25/adapter/'+name,'data');self.write('submissions/mm15_6000/adapters/'+self.m.ARM+'/'+name,'data')
  for gpu,port in ((0,18826),(2,18828)):
   self.write(f'services/gpu{gpu}/health.json','{"passed":true}')
   self.write(f'reports/served-route-effect-gpu{gpu}.json',json.dumps({'passed':True,'adapter_sha256':hashlib.sha256(b'data').hexdigest(),'gpu':gpu,'pid':123+gpu,'create_time':123}))
   self.write(f'services/gpu{gpu}/process.json',json.dumps({'pid':123+gpu,'create_time':123,'physical_gpu_index_at_start':gpu,'port':port,'arm':self.m.ARM,'command':['server','--port',str(port)]}))
   self.write(f'reports/runtime-verification-gpu{gpu}.json',json.dumps({'passed':True,'cuda_graph_captured':True,'requested_content_format':'auto','resolved_content_format':'openai','seed':42,'top_k':-1,'gpu':gpu,'pid':123+gpu,'create_time':123,'adapter_sha256':hashlib.sha256(b'data').hexdigest()}))
  self.write('reports/control-smoke-verification.json','{"passed":true}');self.write('src/budget_contract.py','VERSION="v2"')
 def write(self,name,data):
  p=self.r/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
 def complete(self,*args,**kwargs):
  d=self.r/'runs/full129-gpu0-gpu2/results';d.mkdir(parents=True)
  for i,row in enumerate(self.records):(d/f'{i}.json').write_text(json.dumps(row))
 def test_success_binds_all_sources_and_cleans(self):
  with patch.object(self.m,'assert_identity'),patch.object(self.m,'checked_process',side_effect=self.complete) as run:self.m.run(root=self.r,checkpoint=self.m.CHECKPOINT)
  self.assertEqual(run.call_args.kwargs['timeout'],86400)
  report=json.loads((self.r/'runs/full129-gpu0-gpu2/controller-status.json').read_text())
  self.assertEqual(report['status'],'completed');self.assertEqual(report['summary']['resolved'],1)
  protocol=json.loads((self.r/'runs/full129-gpu0-gpu2/protocol.json').read_text());self.assertIn('src/budget_contract.py',protocol['source_sha256'])
  self.assertEqual(self.svc.cleanup.call_count,2)
  self.assertEqual(protocol['top_k'],-1);self.assertEqual(protocol['seed'],42)
  self.assertEqual([r['resolved_content_format'] for r in protocol['services']],['openai','openai'])
  self.assertEqual(protocol['gpus'],[0,2]);self.assertEqual(protocol['workers'],[2,2])
  command=run.call_args.args[0];start=command.index('--workers');self.assertEqual(command[start+1:start+3],['2','2'])
  self.assertEqual(len(run.call_args.args[3]),2)
  with self.assertRaises(ValueError):self.m.run(root=self.r,checkpoint=self.m.CHECKPOINT)
 def test_subprocess_failure_still_cleans(self):
  with patch.object(self.m,'assert_identity'),patch.object(self.m,'checked_process',side_effect=RuntimeError('failure')):
   with self.assertRaises(RuntimeError):self.m.run(root=self.r,checkpoint=self.m.CHECKPOINT)
  self.assertEqual(self.svc.cleanup.call_count,2)
 def test_cleanup_failure_marks_failure_and_is_raised(self):
  self.svc.cleanup.side_effect=RuntimeError('cleanup failure')
  with patch.object(self.m,'assert_identity'),patch.object(self.m,'checked_process',side_effect=self.complete):
   with self.assertRaises(RuntimeError):self.m.run(root=self.r,checkpoint=self.m.CHECKPOINT)
  report=json.loads((self.r/'runs/full129-gpu0-gpu2/controller-status.json').read_text())
  self.assertEqual(report['status'],'cleanup_failed')
 def test_protocol_rejects_wrong_checkpoint_and_adapter(self):
  self.write('checkpoint-25/COMPLETE.json','{}')
  with self.assertRaises(ValueError):self.m.protocol(self.r,self.gate,{'pid':1,'create_time':1},self.m.CHECKPOINT)
  self.write('checkpoint-25/COMPLETE.json',json.dumps({'step':25,'world_size':2}));self.write('checkpoint-25/adapter/adapter_config.json','other')
  with self.assertRaises(ValueError):self.m.protocol(self.r,self.gate,{'pid':1,'create_time':1},self.m.CHECKPOINT)
 def test_artifact_hashes_and_exception_accounting(self):
  p=self.r/'artifact';p.write_text('x');expected=hashlib.sha256(b'x').hexdigest()
  self.assertEqual(self.m.artifact_hashes([{'path':str(p),'sha256':expected}]),{str(p):expected})
  with self.assertRaises(ValueError):self.m.artifact_hashes({'path':str(p),'sha256':'bad'})
  with self.assertRaises(ValueError):self.m.summarize(self.records[:-1],self.ids[:-1],self.gate)

class ExtraSafetyTests(unittest.TestCase):
 setUp=ControllerTests.setUp
 write=ControllerTests.write
 def test_missing_graph_proof_refuses_evaluation_and_cleans(self):
  self.write('reports/runtime-verification-gpu2.json',json.dumps({'passed':True,'cuda_graph_captured':False,'requested_content_format':'auto','resolved_content_format':'openai','seed':42,'top_k':-1}))
  with patch.object(self.m,'assert_identity'),patch.object(self.m,'checked_process') as run:
   with self.assertRaises(ValueError):self.m.run(root=self.r,checkpoint=self.m.CHECKPOINT)
  run.assert_not_called();self.assertEqual(self.svc.cleanup.call_count,2)

 def test_missing_second_service_proof_fails_closed(self):
  (self.r/'reports/served-route-effect-gpu2.json').unlink()
  with patch.object(self.m,'assert_identity'),patch.object(self.m,'checked_process') as run:
   with self.assertRaises(FileNotFoundError):self.m.run(root=self.r,checkpoint=self.m.CHECKPOINT)
  run.assert_not_called();self.assertEqual(self.svc.cleanup.call_count,2)
 def test_first_cleanup_failure_does_not_skip_second_owned_service(self):
  self.svc.cleanup.side_effect=[RuntimeError('failure'),{'terminated':[]}]
  with patch.object(self.m,'assert_identity'),patch.object(self.m,'checked_process',side_effect=ControllerTests.complete.__get__(self)):
   with self.assertRaises(RuntimeError):self.m.run(root=self.r,checkpoint=self.m.CHECKPOINT)
  self.assertEqual(self.svc.cleanup.call_count,2)

 def test_stale_proof_cannot_be_reused_for_another_process(self):
  p=self.r/'reports/served-route-effect-gpu2.json';d=json.loads(p.read_text());p.write_text(json.dumps({**d,'pid':999}))
  with patch.object(self.m,'assert_identity'),patch.object(self.m,'checked_process') as run:
   with self.assertRaisesRegex(ValueError,'different service'):self.m.run(root=self.r,checkpoint=self.m.CHECKPOINT)
  run.assert_not_called();self.assertEqual(self.svc.cleanup.call_count,2)

if __name__=='__main__':unittest.main()
