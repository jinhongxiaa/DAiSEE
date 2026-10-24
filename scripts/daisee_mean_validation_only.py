#!/usr/bin/env python3
"""Revision helper: reproduce mean-feature search without evaluating Test.
Not an original historical script. Uses the loop/grid/tie rule in the archived tuned script.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support
TASKS=('Boredom','Engagement','Confusion','Frustration')
def search(data,grid,seed):
 tr=data['split']=='Train';va=data['split']=='Validation'
 if not tr.any() or not va.any():raise ValueError('Train and Validation required')
 rows=[];selected=[]
 for task in TASKS:
  candidates=[]
  for c in sorted(set(grid)):
   for cw in ['balanced',None]:
    model=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(C=c,class_weight=cw,max_iter=5000,random_state=seed))]);model.fit(data['X'][tr],data[task][tr]);y=data[task][va];p=model.predict(data['X'][va])
    q,r,_,_=precision_recall_fscore_support(y,p,labels=[0,1,2,3],average='macro',zero_division=0)
    row=dict(task=task,C=c,class_weight=cw or 'none',macro_f1=f1_score(y,p,labels=[0,1,2,3],average='macro',zero_division=0),weighted_f1=f1_score(y,p,labels=[0,1,2,3],average='weighted',zero_division=0),accuracy=accuracy_score(y,p),macro_precision=q,macro_recall=r);rows.append(row);candidates.append(row)
  selected.append(max(candidates,key=lambda z:(z['macro_f1'],z['weighted_f1'],-z['C'])))
 return rows,selected
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--c-grid',default='0.01,0.1,1,10,100');p.add_argument('--seed',type=int,default=20260928);a=p.parse_args();grid=[float(v) for v in a.c_grid.split(',')]
 if not grid or any(v<=0 for v in grid):p.error('C must be positive')
 with np.load(a.features,allow_pickle=False) as z:rows,selected=search(z,grid,a.seed)
 a.output.mkdir(parents=True,exist_ok=True)
 for name,content in [('validation_search.csv',rows),('selected_mean_candidates.csv',selected)]:
  with (a.output/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=content[0]);w.writeheader();w.writerows(content)
 (a.output/'revision_helper_record.json').write_text(json.dumps(dict(status='new revision helper, not historical pre-Test evidence',seed=a.seed,grid=grid,test_predictions_computed=False),indent=2))
if __name__=='__main__':main()
