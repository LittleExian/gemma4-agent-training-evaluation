import json,tempfile,unittest
from pathlib import Path
from provenance import check_resume_manifest
from serialization import SERIALIZATION_VERSION
class ProvenanceTests(unittest.TestCase):
 def test_missing_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):check_resume_manifest(Path(d)/'checkpoint-1',{})
 def test_changed_objective_and_data_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   manifest={'serialization_version':SERIALIZATION_VERSION,'data_sha256':'a','objective':'token_mean','model':'m','revision':'r',
             'world_size':2,'accum':4,'lr':.01,'seed':42,'source_sha256':{'train.py':'s'}}
   (Path(d)/'training_manifest.json').write_text(json.dumps(manifest))
   check_resume_manifest(Path(d)/'checkpoint-1',manifest)
   for k,v in [('serialization_version','legacy'),('objective','task_mean'),('data_sha256','b'),('source_sha256',{})]:
    with self.assertRaises(ValueError):check_resume_manifest(Path(d)/'checkpoint-1',dict(manifest,**{k:v}))
if __name__=='__main__':unittest.main()
