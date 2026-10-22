import tempfile
from pathlib import Path
import unittest
import gemmacodeagent_full_task as t

class TransformTests(unittest.TestCase):
    def test_replacements_fail_closed(self):
        with self.assertRaises(ValueError): t.replace_once('x x','x','y')
        with self.assertRaises(ValueError): t.replace_once('x','missing','y')
        self.assertEqual(t.replace_once('xx anchor yy','anchor','ok'),'xx ok yy')
    def test_endpoint_is_loopback_only(self):
        self.assertEqual(t.validate_endpoint('http://127.0.0.1:8200/v1'),'http://127.0.0.1:8200/v1')
        for endpoint in ('https://example.org/v1','http://user:secret@127.0.0.1:8106/v1','http://127.0.0.1:8106/v1?secret=token'):
            with self.assertRaises(ValueError): t.validate_endpoint(endpoint)
    def test_error_is_not_a_normal_test_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            error={'task_id':'x','status':'ERROR','resolved':False,'test_exit_code':0,
                   'error_message':'APIConnectionError: connection refused'}
            data=t.normalize_result(error,Path(folder),{},[],'fp','agent')
            self.assertTrue(data['infra_error'])
            self.assertFalse(data['agent_timed_out'])
    def test_normalization_keeps_controls_private_and_patch_stable(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            row={'resolved':False,'status':'TIMEOUT','agent_patch':'diff x','test_exit_code':1,'test_output':'1 failed'}
            data=t.normalize_result(row,out,{'profile_id':'p'},[],'fingerprint','agent')
            self.assertTrue(data['agent_timed_out'])
            self.assertEqual(Path(data['patch_file']).read_text(),'diff x')
            self.assertNotIn('agent_patch',data)
            self.assertEqual(data['verification']['test_exit_code'],1)
            self.assertEqual(data['dependency_profile']['profile_id'],'p')

if __name__=='__main__': unittest.main()
