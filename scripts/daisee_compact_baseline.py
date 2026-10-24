#!/usr/bin/env python3
r"""Compact, reproducible DAiSEE affect-recognition baseline.

The script samples a fixed number of frames per authorised local video, pools
pretrained ResNet-18 embeddings, and fits one balanced multinomial logistic
regression model for each original DAiSEE affect dimension. It respects the
official Train/Validation/Test assignment in the compact manifest.

Examples (PowerShell):
  py .\daisee_compact_baseline.py --manifest .\DAiSEE_Compact\DAiSEE_Compact_manifest.csv \
    --video-root .\DAiSEE_Compact\videos --output .\results --stage extract
  py .\daisee_compact_baseline.py --manifest .\DAiSEE_Compact\DAiSEE_Compact_manifest.csv \
    --video-root .\DAiSEE_Compact\videos --output .\results --stage train
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

TASKS = ("Boredom", "Engagement", "Confusion", "Frustration")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(handle)]
    required = {"OfficialSplit", "ClipID", *TASKS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must include {sorted(required)}")
    return rows


def locate_videos(rows: list[dict[str, str]], root: Path) -> dict[tuple[str, str], Path]:
    wanted = {(row["OfficialSplit"], row["ClipID"]) for row in rows}
    found: dict[tuple[str, str], Path] = {}
    for split in ("Train", "Validation", "Test"):
        directory = root / split
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing video split directory: {directory}")
        for path in directory.rglob("*"):
            key = (split, path.name)
            if path.is_file() and key in wanted:
                found[key] = path
    missing = wanted - set(found)
    if missing:
        example = ", ".join(f"{split}/{clip}" for split, clip in sorted(missing)[:5])
        raise FileNotFoundError(f"Could not locate {len(missing)} selected videos; e.g. {example}")
    return found


def sampled_frames(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_total <= 0:
        capture.release()
        raise RuntimeError(f"Unreadable video: {path}")
    indices = np.linspace(0, frame_total - 1, count, dtype=int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No usable frames: {path}")
    return frames


def extract(rows: list[dict[str, str]], videos: dict[tuple[str, str], Path], output: Path,
            frames_per_clip: int, batch_size: int, device: str) -> Path:
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    network = resnet18(weights=weights)
    encoder = nn.Sequential(*list(network.children())[:-1]).to(device).eval()
    vectors, retained = [], []
    with torch.inference_mode():
        for position, row in enumerate(rows, start=1):
            path = videos[(row["OfficialSplit"], row["ClipID"])]
            frames = sampled_frames(path, frames_per_clip)
            frame_tensors = [transform(torch.from_numpy(frame).permute(2, 0, 1)) for frame in frames]
            chunks = []
            for start in range(0, len(frame_tensors), batch_size):
                batch = torch.stack(frame_tensors[start:start + batch_size]).to(device)
                chunks.append(encoder(batch).flatten(1).cpu().numpy())
            vectors.append(np.concatenate(chunks, axis=0).mean(axis=0))
            retained.append(row)
            print(f"[{position:04d}/{len(rows):04d}] {row['OfficialSplit']}/{row['ClipID']}", flush=True)
    feature_path = output / "resnet18_mean_embeddings.npz"
    np.savez_compressed(
        feature_path,
        X=np.asarray(vectors, dtype=np.float32),
        clip_id=np.asarray([row["ClipID"] for row in retained]),
        split=np.asarray([row["OfficialSplit"] for row in retained]),
        **{task: np.asarray([int(row[task]) for row in retained], dtype=np.int64) for task in TASKS},
    )
    return feature_path


def train(feature_path: Path, output: Path, seed: int) -> None:
    data = np.load(feature_path, allow_pickle=False)
    X, splits = data["X"], data["split"]
    train_mask, validation_mask, test_mask = splits == "Train", splits == "Validation", splits == "Test"
    if not (train_mask.any() and validation_mask.any() and test_mask.any()):
        raise ValueError("Embeddings must contain official Train, Validation and Test samples")
    records, prediction_rows = [], []
    for task in TASKS:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ])
        model.fit(X[train_mask], data[task][train_mask])
        for split_name, mask in (("Validation", validation_mask), ("Test", test_mask)):
            truth, predicted = data[task][mask], model.predict(X[mask])
            precision, recall, _, _ = precision_recall_fscore_support(
                truth, predicted, average="macro", zero_division=0
            )
            records.append({
                "task": task, "split": split_name, "n": int(mask.sum()),
                "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
                "weighted_f1": f1_score(truth, predicted, average="weighted", zero_division=0),
                "accuracy": accuracy_score(truth, predicted), "macro_precision": precision, "macro_recall": recall,
            })
            for clip_id, y_true, y_pred in zip(data["clip_id"][mask], truth, predicted):
                prediction_rows.append({"task": task, "split": split_name, "ClipID": str(clip_id), "true": int(y_true), "predicted": int(y_pred)})
            if split_name == "Test":
                matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2, 3])
                figure, axis = plt.subplots(figsize=(4, 3.5))
                image = axis.imshow(matrix, cmap="Blues")
                axis.set(xticks=range(4), yticks=range(4), xlabel="Predicted label", ylabel="True label", title=f"{task}: Test")
                for i in range(4):
                    for j in range(4):
                        axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
                figure.colorbar(image, ax=axis)
                figure.tight_layout()
                figure.savefig(output / f"confusion_{task.lower()}.png", dpi=200)
                plt.close(figure)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("extract", "train", "all"), default="all")
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.frames_per_clip < 1: parser.error("--frames-per-clip must be positive")
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    feature_path = args.output / "resnet18_mean_embeddings.npz"
    rows = load_manifest(args.manifest)
    if args.stage in ("extract", "all"):
        videos = locate_videos(rows, args.video_root)
        feature_path = extract(rows, videos, args.output, args.frames_per_clip, args.batch_size, args.device)
    if args.stage in ("train", "all"):
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing features: {feature_path}; run with --stage extract first")
        train(feature_path, args.output, args.seed)
    (args.output / "run_metadata.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "frames_per_clip": args.frames_per_clip, "device": args.device,
        "protocol": "ResNet-18 mean frame embeddings + balanced multinomial logistic regression",
        "result_status": "generated locally; inspect metrics.csv before reporting",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
