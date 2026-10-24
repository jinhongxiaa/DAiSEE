"""Synthetic checks only; no DAiSEE download or weights required."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from daisee_mean_validation_only import search,TASKS
from daisee_temporal_validation import profile
from build_daisee_compact import allocate_strata
class Smoke(unittest.TestCase):
 def test_test_isolation(self):
  rng=np.random.default_rng(97);d={'X':rng.normal(size=(48,5)),'split':np.array(['Train']*24+['Validation']*12+['Test']*12)}
  for t in TASKS:d[t]=np.arange(48)%4
  before=search(d,[.01,.1],7);changed={k:v.copy() for k,v in d.items()};changed['X'][36:]=10000
  for t in TASKS:changed[t][36:]=3-changed[t][36:]
  self.assertEqual(before,search(changed,[.01,.1],7))
 def test_profile(self):
  x=np.array([[[1.,2.],[3.,6.],[5.,10.]]]);got=profile(x)
  np.testing.assert_allclose(got,np.concatenate([x.mean(1),x.std(1),[[4.,8.]],[[2.,4.]]],axis=1))
 def test_subset(self):
  rows=[dict(ClipID=f'{i:06d}.avi',SubjectID=str(i//2),OfficialSplit='Train',**{t:str(i%4) for t in TASKS}) for i in range(32)]
  a=allocate_strata(rows,12,20260928);b=allocate_strata(rows,12,20260928)
  self.assertEqual(a,b);self.assertEqual(len(a),12);self.assertEqual(len({x['ClipID'] for x in a}),12)
if __name__=='__main__':unittest.main()
