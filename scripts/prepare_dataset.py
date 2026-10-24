"""Generate a local compact subset from authorized DAiSEE labels."""
import argparse,hashlib,subprocess,sys
from pathlib import Path
EXPECTED='bca66a705e3f1a744bcd8ff97ad6893d4e92135e98055dd570e20ceacb848125'
def main():
 p=argparse.ArgumentParser();p.add_argument('--labels-dir',type=Path,default=Path('data/DAiSEE/Labels'));p.add_argument('--out-dir',type=Path,default=Path('data/compact'));a=p.parse_args()
 cmd=[sys.executable,str(Path(__file__).with_name('build_daisee_compact.py')),'--labels-dir',str(a.labels_dir),'--out-dir',str(a.out_dir),'--seed','20260928']
 subprocess.run(cmd,check=True)
 path=a.out_dir/'DAiSEE_Compact_manifest.csv';digest=hashlib.sha256(path.read_bytes()).hexdigest()
 print('Manifest SHA-256:',digest)
 if digest!=EXPECTED:
  raise SystemExit('Manifest differs from the archived study. Do not report an exact reproduction. Check source label release and CSV content; keep this output for provenance.')
 print('ARCHIVED_MANIFEST_MATCH')
if __name__=='__main__':main()
