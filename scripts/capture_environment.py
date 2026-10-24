#!/usr/bin/env python3
"""Record an actual current environment without claiming it is the original one."""
import argparse,datetime,hashlib,importlib.metadata as md,json,platform
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--weights',type=Path);a=p.parse_args();versions={}
for name in ['numpy','pandas','matplotlib','scipy','scikit-learn','torch','torchvision','opencv-python','Pillow']:
 try:versions[name]=md.version(name)
 except md.PackageNotFoundError:versions[name]=None
r=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=platform.python_version(),platform=platform.platform(),packages=versions,status='environment at capture time; not retrospective proof of original versions')
if a.weights:r['weights_file']=a.weights.name;r['weights_sha256']=hashlib.sha256(a.weights.read_bytes()).hexdigest()
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
