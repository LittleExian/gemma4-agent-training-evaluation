"""Numerical guard for the collaborator's archived memory optimization."""
import json,torch
import torch.nn.functional as F
from weighted_lora_streamed_ce import weighted_sum

def reference(logits,labels,weights):
 total=logits.new_zeros((),dtype=torch.float32)
 for i in range(0,len(labels),128):
  total=total+(F.cross_entropy(logits[0,i:i+128].float(),labels[i:i+128],reduction='none')*weights[i:i+128]).sum()
 return total
checks=0
torch.set_num_threads(1)
for dtype in [torch.float32,torch.bfloat16]:
 for length in [1,127,128,129,257]:
  for upstream in [.001,1.,3.1567]:
   torch.manual_seed(42);x=torch.randn(1,length,37,dtype=dtype,requires_grad=True);y=x.detach().clone().requires_grad_()
   labels=torch.randint(37,(length,));weights=torch.rand(length)
   a=reference(x,labels,weights);b=weighted_sum(y,labels,weights)
   (a*upstream).backward();(b*upstream).backward()
   torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0);checks+=1
print(json.dumps({'passed':True,'checks':checks,'device':'CPU','loss_and_first_order_gradients_exact':True}))
