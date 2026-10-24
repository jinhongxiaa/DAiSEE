#!/usr/bin/env python3
"""Validate original manifest, prediction labels and archived Table 1 scores."""
import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--metrics',type=Path,required=True);a=p.parse_args()
m=pd.read_csv(a.manifest,dtype={'ClipID':str,'SubjectID':str});pred=pd.read_csv(a.predictions,dtype={'ClipID':str});metrics=pd.read_csv(a.metrics).set_index('task')
if len(m)!=960 or m.ClipID.duplicated().any():raise ValueError('Invalid manifest')
if m.groupby('OfficialSplit').size().to_dict()!={'Test':200,'Train':600,'Validation':160}:raise ValueError('Split count mismatch')
groups={s:set(g.SubjectID) for s,g in m.groupby('OfficialSplit')}
for s in groups:
 for t in groups:
  if s!=t and groups[s]&groups[t]:raise ValueError('Participant overlap')
if set(pred.task)!={'Boredom','Engagement','Confusion','Frustration'} or len(pred)!=800:raise ValueError('Task coverage mismatch')
for task,g in pred.groupby('task'):
 z=g.merge(m[m.OfficialSplit=='Test'][['ClipID',task]],on='ClipID',validate='one_to_one')
 if len(z)!=200 or not (z.true==z[task]).all():raise ValueError('Prediction/manifest mismatch')
 score=f1_score(z.true,z.predicted,labels=[0,1,2,3],average='macro',zero_division=0)
 if abs(score-metrics.loc[task,'macro_f1'])>1e-12:raise ValueError('Archived metric mismatch')
print('ARCHIVED_ARTIFACT_CHECK_PASS: counts, participant IDs, label alignment and macro-F1')
