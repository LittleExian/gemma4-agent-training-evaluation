import unittest
from objectives import task_scales, scalar_loss, validate_rows
class ObjectiveTests(unittest.TestCase):
 def test_exact_task_average(self):
  rows=[{'task_id':'a'},{'task_id':'b'},{'task_id':'b'},{'task_id':'b'}]
  scales=task_scales(rows)
  numerators=[8,2,12,40];weights=[2,1,3,5]
  actual=sum(scalar_loss(n,w,scales[r['task_id']],'task_mean') for r,n,w in zip(rows,numerators,weights))/4
  self.assertAlmostEqual(actual,(4+(2+4+8)/3)/2)
 def test_token_objective_unchanged(self):self.assertEqual(scalar_loss(8,2,3,'token_mean'),8)
 def test_invalid_weights(self):
  r={'sample_id':'a','input_ids':[1,2],'weights':[0,float('nan')]}
  with self.assertRaises(ValueError):validate_rows([r])
 def test_no_supervision(self):
  with self.assertRaises(ValueError):validate_rows([{'sample_id':'a','input_ids':[1,2],'weights':[1,0]}])
if __name__=='__main__':unittest.main()
