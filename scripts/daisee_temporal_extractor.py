#!/usr/bin/env python3
r"""Extract ordered ResNet-18 frame embeddings for temporal validation.

Unlike the mean-embedding baseline, this file preserves the six embeddings
per clip. It writes no predictions and uses no labels for model selection.

Example:
  python .\daisee_temporal_extractor.py --manifest .\DAiSEE_Compact_manifest.csv --video-root .\videos --output .\temporal_features --frames-per-clip 6 --device cpu --batch-size 4
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import cv2, numpy as np, torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

TASKS = ("Boredom", "Engagement", "Confusion", "Frustration")

def manifest(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows or not {"OfficialSplit", "ClipID", *TASKS}.issubset(rows[0]):
        raise ValueError("Manifest lacks required DAiSEE columns")
    return rows

def locate(rows, root: Path):
    wanted = {(r["OfficialSplit"], r["ClipID"]) for r in rows}; found = {}
    for split in ("Train", "Validation", "Test"):
        folder = root / split
        if not folder.is_dir(): raise FileNotFoundError(f"Missing {folder}")
        for path in folder.rglob("*"):
            if path.is_file() and (split, path.name) in wanted: found[(split, path.name)] = path
    if missing := wanted - set(found): raise FileNotFoundError(f"Missing {len(missing)} authorised videos")
    return found

def frames(path: Path, n: int):
    cap = cv2.VideoCapture(str(path)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: cap.release(); raise RuntimeError(f"Unreadable video: {path}")
    output = []
    for index in np.linspace(0, total - 1, n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index)); ok, frame = cap.read()
        if not ok or frame is None: cap.release(); raise RuntimeError(f"Unreadable frame in {path}")
        output.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release(); return output

def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--frames-per-clip", type=int, default=6); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = p.parse_args()
    if a.frames_per_clip < 2: p.error("--frames-per-clip must be at least 2")
    a.output.mkdir(parents=True, exist_ok=True); rows = manifest(a.manifest); videos = locate(rows, a.video_root)
    weights = ResNet18_Weights.DEFAULT; transform = weights.transforms(); encoder = nn.Sequential(*list(resnet18(weights=weights).children())[:-1]).to(a.device).eval()
    sequences = []
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            tensors = [transform(torch.from_numpy(x).permute(2, 0, 1)) for x in frames(videos[(row["OfficialSplit"], row["ClipID"])], a.frames_per_clip)]
            chunks = [encoder(torch.stack(tensors[i:i+a.batch_size]).to(a.device)).flatten(1).cpu().numpy() for i in range(0, len(tensors), a.batch_size)]
            sequences.append(np.concatenate(chunks)); print(f"[{number:04d}/{len(rows):04d}] {row['OfficialSplit']}/{row['ClipID']}", flush=True)
    np.savez_compressed(a.output / "resnet18_sequence_embeddings.npz", X=np.asarray(sequences, dtype=np.float32),
                        clip_id=np.asarray([r["ClipID"] for r in rows]), split=np.asarray([r["OfficialSplit"] for r in rows]),
                        **{t: np.asarray([int(r[t]) for r in rows], dtype=np.int64) for t in TASKS})

if __name__ == "__main__": main()
