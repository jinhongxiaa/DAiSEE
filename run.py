"""Run supported stages with a conservative CPU budget. Run from repository root."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract-mean','extract-temporal','validate-mean','validate-temporal','evaluate']);p.add_argument('--video-root',default='data/DAiSEE/DataSet');a=p.parse_args()
 os.chdir(ROOT)
 try:
  import torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
 except ImportError:pass
 common=['--manifest','data/compact/DAiSEE_Compact_manifest.csv','--video-root',a.video_root,'--device','cpu','--frames-per-clip','6','--batch-size','4']
 jobs={
 'extract-mean':('daisee_compact_baseline.py',common+['--output','outputs/mean','--stage','extract']),
 'extract-temporal':('daisee_temporal_extractor.py',common+['--output','outputs/temporal']),
 'validate-mean':('daisee_mean_validation_only.py',['--features','outputs/mean/resnet18_mean_embeddings.npz','--output','outputs/mean_validation']),
 'validate-temporal':('daisee_temporal_validation.py',['--features','outputs/temporal/resnet18_sequence_embeddings.npz','--output','outputs/temporal_validation']),
 'evaluate':('daisee_final_selected_evaluation.py',['--mean-features','outputs/mean/resnet18_mean_embeddings.npz','--temporal-features','outputs/temporal/resnet18_sequence_embeddings.npz','--output','outputs/retrospective_final'])}
 script,args=jobs[a.stage];out=ROOT/args[args.index('--output')+1]
 if out.exists() and any(out.iterdir()):raise SystemExit(f'Output directory is not empty: {out}. Preserve or move it before rerunning.')
 sys.path.insert(0,str(ROOT/'scripts'));sys.argv=[script,*args];runpy.run_path(str(ROOT/'scripts'/script),run_name='__main__')
if __name__=='__main__':main()
