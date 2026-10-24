#!/usr/bin/env python3
"""Audit archived validation tables; no feature loading or model training."""
import argparse,csv,json
from pathlib import Path
TASKS=['Boredom','Engagement','Confusion','Frustration']
def read(path):
 with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def choose(mean,temp):
 rows=[]
 for task in TASKS:
  m=max((r for r in mean if r['task']==task),key=lambda r:(float(r['macro_f1']),float(r['weighted_f1']),-float(r['C'])))
  t=max((r for r in temp if r['task']==task),key=lambda r:(float(r['macro_f1']),float(r['weighted_f1'])))
  # The cross-representation tie rule was not documented historically.
  tie=(float(t['macro_f1']),float(t['weighted_f1']))==(float(m['macro_f1']),float(m['weighted_f1']))
  if tie:raise ValueError('Cross-representation tie: historical rule is undocumented')
  rep='temporal' if (float(t['macro_f1']),float(t['weighted_f1']))>(float(m['macro_f1']),float(m['weighted_f1'])) else 'mean';best=t if rep=='temporal' else m
  rows.append(dict(task=task,representation=rep,C=float(best['C']),class_weight=best.get('class_weight','balanced'),mean_validation_f1=float(m['macro_f1']),temporal_validation_f1=float(t['macro_f1'])))
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--mean-table',type=Path,required=True);p.add_argument('--temporal-table',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=choose(read(a.mean_table),read(a.temporal_table));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='retrospective replay of archived validation tables; not a historical freeze record',selection=rows),indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
