#!/usr/bin/env python3
r"""Retrospective reproduction of fixed historical candidates.
Historical Test predictions existed before the recovered final run.
Legacy clip-stratified CIs are not participant-cluster uncertainty intervals.
"""
from __future__ import annotations
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

CHOICE={"Boredom":("mean",10.0),"Engagement":("mean",0.1),"Confusion":("mean",0.1),"Frustration":("temporal",1.0)}
def temporal(x): return np.concatenate((x.mean(1),x.std(1),x[:,-1]-x[:,0],np.abs(np.diff(x,axis=1)).mean(1)),axis=1)
def measures(y,p):
 q,r,_,_=precision_recall_fscore_support(y,p,average="macro",labels=[0,1,2,3],zero_division=0)
 return {"macro_f1":f1_score(y,p,average="macro",labels=[0,1,2,3],zero_division=0),"weighted_f1":f1_score(y,p,average="weighted",labels=[0,1,2,3],zero_division=0),"accuracy":accuracy_score(y,p),"macro_precision":q,"macro_recall":r}
def ci(y,p,seed):
 rng=np.random.default_rng(seed); groups=[np.flatnonzero(y==c) for c in range(4)]; a=[]
 for _ in range(1000):
  ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups if len(g)])
  a.append(f1_score(y[ix],p[ix],average="macro",labels=[0,1,2,3],zero_division=0))
 return np.percentile(a,[2.5,97.5])
def main():
 p=argparse.ArgumentParser();p.add_argument("--mean-features",type=Path,required=True);p.add_argument("--temporal-features",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,default=20260928);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 mean=np.load(a.mean_features,allow_pickle=False); seq=np.load(a.temporal_features,allow_pickle=False)
 if not np.array_equal(mean["clip_id"],seq["clip_id"]) or not np.array_equal(mean["split"],seq["split"]):raise ValueError("Feature files are not aligned")
 
 for task in CHOICE:
  if not np.array_equal(mean[task],seq[task]):raise ValueError("Feature labels are not aligned")
 if set(mean['split']) != {'Train','Validation','Test'}:raise ValueError('Expected exactly Train/Validation/Test')
 tr=np.isin(mean["split"],["Train","Validation"]);te=mean["split"]=="Test"; rows=[];predrows=[]
 for task,(kind,c) in CHOICE.items():
  X=mean["X"] if kind=="mean" else temporal(seq["X"]); n=min(128,int(tr.sum())-1,X.shape[1]) if kind=="temporal" else None
  steps=[("scale",StandardScaler())];
  if n: steps.append(("pca",PCA(n_components=n,whiten=True,random_state=a.seed)))
  steps.append(("classifier",LogisticRegression(C=c,class_weight="balanced",max_iter=5000,random_state=a.seed)))
  model=Pipeline(steps);model.fit(X[tr],mean[task][tr]);y,pred=mean[task][te],model.predict(X[te]);low,high=ci(y,pred,a.seed+sum(map(ord,task)))
  rows.append({"task":task,"representation":kind,"C":c,"n_test":int(te.sum()),**measures(y,pred),"macro_f1_ci_low":low,"macro_f1_ci_high":high})
  predrows.extend({"task":task,"ClipID":str(i),"true":int(y0),"predicted":int(p0)} for i,y0,p0 in zip(mean["clip_id"][te],y,pred))
  m=confusion_matrix(y,pred,labels=[0,1,2,3]);fig,ax=plt.subplots(figsize=(4,3.5));im=ax.imshow(m,cmap="Blues");ax.set(xticks=range(4),yticks=range(4),xlabel="Predicted label",ylabel="True label",title=f"{task}: Test")
  for i in range(4):
   for j in range(4):ax.text(j,i,str(m[i,j]),ha="center",va="center")
  fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(a.output/f"confusion_{task.lower()}.png",dpi=200);plt.close(fig)
 for name,content in (("final_selected_metrics.csv",rows),("final_selected_predictions.csv",predrows)):
  with (a.output/name).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(content[0]));w.writeheader();w.writerows(content)
 (a.output/"final_protocol.json").write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),"selection_source":"Validation only","fixed_choice":CHOICE,"evaluation_status":"retrospective_reproduction", "historical_test_evaluation_count":"at_least_two_sets", "interval_method":"legacy_class_stratified_clip_bootstrap_not_participant_cluster"},indent=2),encoding="utf-8")
if __name__=="__main__":main()
