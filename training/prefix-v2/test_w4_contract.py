"""CPU guards: deployment identity, resume rejection, and signed INT4 decode."""
import json,tempfile,unittest
from pathlib import Path
import torch
from base_loading import MODEL_ID,MODEL_REVISION,base_identity,check_source,check_resume,validate_quant_config
from validate_w4 import unpack_rows

class ContractTest(unittest.TestCase):
 def test_identity_and_resume(self):
  check_source(MODEL_ID,MODEL_REVISION)
  with self.assertRaises(ValueError):check_source('google/gemma-4-31B-it-qat-q4_0-unquantized',MODEL_REVISION)
  with self.assertRaises(ValueError):check_source(MODEL_ID,'main')
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):check_resume(d)
   p=Path(d)/'base_identity.json';p.write_text(json.dumps(base_identity()));check_resume(d)
   x=base_identity();x['base_loading']='old';p.write_text(json.dumps(x))
   with self.assertRaises(ValueError):check_resume(d)
 def test_wrong_quantization(self):
  with self.assertRaises(ValueError):validate_quant_config({})
  w=dict(num_bits=4,type='int',group_size=32,symmetric=True,strategy='group')
  c={'quantization_config':{'quant_method':'compressed-tensors','format':'pack-quantized','config_groups':{'g':{'weights':w}}}}
  validate_quant_config(c);w['group_size']=128
  with self.assertRaises(ValueError):validate_quant_config(c)
 def test_decode_signed_extremes(self):
  from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32,unpack_from_int32
  from compressed_tensors.quantization.lifecycle.forward import dequantize
  q=torch.arange(-8,8,dtype=torch.int8).repeat(8).reshape(2,64)
  scale=torch.tensor([[.031,.079],[.067,.125]],dtype=torch.bfloat16)
  packed=pack_to_int32(q,4)
  reference=dequantize(unpack_from_int32(packed,4,q.shape),scale)
  torch.testing.assert_close(unpack_rows(packed,scale,64),reference,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
