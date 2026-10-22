import unittest
import gemmacodeagent_full_adapter as a

class ProfileTests(unittest.TestCase):
    def test_strict_normal_profile(self):
        profile={'profile_id':'p','environment':'/env','actual_freeze':'pytest==8\n'}
        gate={'eligible_ids':['x'],'task_bindings':{'x':{'profile':profile}}}
        self.assertEqual(a.select_profile(gate,'x',None,'pytest==8\n'),profile)
        for task,pid,freeze in [('y',None,'pytest==8\n'),('x','other','pytest==8\n'),('x',None,'pytest==9\n')]:
            with self.assertRaises(ValueError): a.select_profile(gate,task,pid,freeze)
    def test_matrix_profile_must_be_required_and_frozen(self):
        p={'profile_id':'p','environment':'/env','actual_freeze':'x==1\n'}
        gate={'eligible_ids':['x'],'task_bindings':{'x':{'required_profile_ids':['p','q'], 'matrix':{'arms':[{'profile_id':'p','dependency_profile':p}]}}}}
        self.assertEqual(a.select_profile(gate,'x','p','x==1\n'),p)
        for pid in (None,'q','z'):
            with self.assertRaises(ValueError): a.select_profile(gate,'x',pid,'x==1\n')

if __name__=='__main__': unittest.main()
