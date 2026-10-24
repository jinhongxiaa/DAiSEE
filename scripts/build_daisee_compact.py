#!/usr/bin/env python3
"""Create a deterministic, compact DAiSEE video subset.

This script selects clips only from DAiSEE's official Train, Validation and
Test splits.  It does not alter a split or move a subject across splits.
The selection is stratified on the four supplied ordinal labels jointly, so
rare affect combinations are retained for a small pilot-scale study.

Example:
  python build_daisee_compact.py --labels-dir Labels --out-dir DAiSEE_Compact
  python build_daisee_compact.py --labels-dir Labels --out-dir DAiSEE_Compact \
      --dataset-root Dataset --copy-videos

The source videos remain private.  Do not redistribute the created video
directory; distribute this script and manifest instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

FIELDS = ("Boredom", "Engagement", "Confusion", "Frustration")
SPLITS = (
    ("Train", "TrainLabels.csv", 600),
    ("Validation", "ValidationLabels.csv", 160),
    ("Test", "TestLabels.csv", 200),
)


def stable_key(clip_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest()


def read_rows(path: Path, split: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            clean = {(key or "").strip(): (value or "").strip() for key, value in row.items()}
            if not clean.get("ClipID") or any(clean.get(field, "") not in {"0", "1", "2", "3"} for field in FIELDS):
                raise ValueError(f"Invalid DAiSEE row in {path.name}: {row}")
            clean["OfficialSplit"] = split
            clean["SubjectID"] = clean["ClipID"].split(".")[0][:6]
            rows.append(clean)
    return rows


def allocate_strata(rows: list[dict[str, str]], budget: int, seed: int) -> list[dict[str, str]]:
    strata: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[tuple(row[field] for field in FIELDS)].append(row)
    if budget > len(rows):
        raise ValueError("Subset budget exceeds the available clips")
    # Retain one clip per observed joint label combination, then distribute the
    # remaining budget proportionally across the remaining clips (largest
    # remainder allocation).  This retains rare combinations without turning
    # the subset into a fully balanced, artificial distribution.
    ordered = sorted(strata.items())
    base = {key: 1 for key, _ in ordered} if budget >= len(ordered) else {key: 0 for key, _ in ordered}
    remaining = budget - sum(base.values())
    capacity = {key: len(items) - base[key] for key, items in ordered}
    total_capacity = sum(capacity.values())
    allocation = dict(base)
    if remaining:
        shares = {key: remaining * capacity[key] / total_capacity for key, _ in ordered}
        for key, _ in ordered:
            add = min(capacity[key], int(shares[key]))
            allocation[key] += add
        leftovers = budget - sum(allocation.values())
        ranked = sorted(ordered, key=lambda pair: (shares[pair[0]] % 1, len(pair[1]), pair[0]), reverse=True)
        while leftovers:
            progressed = False
            for key, items in ranked:
                if allocation[key] < len(items):
                    allocation[key] += 1
                    leftovers -= 1
                    progressed = True
                    if not leftovers:
                        break
            if not progressed:
                raise RuntimeError("Could not complete stratum allocation")

    selected: list[dict[str, str]] = []
    for key, items in ordered:
        # A subject round-robin prevents a common label combination from being
        # concentrated in only a few students.  Its order is seed-controlled.
        by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
        for item in items:
            by_subject[item["SubjectID"]].append(item)
        subject_lists = []
        for subject, subject_items in by_subject.items():
            subject_lists.append((stable_key(subject, seed), sorted(subject_items, key=lambda x: stable_key(x["ClipID"], seed))))
        subject_lists.sort()
        pool: list[dict[str, str]] = []
        depth = 0
        while len(pool) < len(items):
            added = False
            for _, subject_items in subject_lists:
                if depth < len(subject_items):
                    pool.append(subject_items[depth])
                    added = True
            if not added:
                break
            depth += 1
        selected.extend(pool[: allocation[key]])
    return sorted(selected, key=lambda row: (row["OfficialSplit"], row["ClipID"]))


def split_directory(root: Path, split: str) -> Path:
    names = {
        "Train": ("Train", "Training"),
        "Validation": ("Validation",),
        "Test": ("Test", "Testing"),
    }[split]
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No {split} directory under {root}")


def copy_videos(rows: list[dict[str, str]], dataset_root: Path, output_root: Path) -> None:
    for split, _, _ in SPLITS:
        source_split = split_directory(dataset_root, split)
        wanted = {row["ClipID"] for row in rows if row["OfficialSplit"] == split}
        found = 0
        for video in source_split.rglob("*"):
            if video.is_file() and video.name in wanted:
                destination = output_root / "videos" / split / video.relative_to(source_split)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video, destination)
                found += 1
        missing = wanted - {path.name for path in (output_root / "videos" / split).rglob("*") if path.is_file()} if (output_root / "videos" / split).exists() else wanted
        if missing:
            raise FileNotFoundError(f"{split}: {len(missing)} selected clips were not found, e.g. {sorted(missing)[:5]}")
        print(f"Copied {found} {split} videos")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--copy-videos", action="store_true")
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    if args.copy_videos and not args.dataset_root:
        parser.error("--copy-videos requires --dataset-root")

    selected: list[dict[str, str]] = []
    for split, filename, budget in SPLITS:
        rows = read_rows(args.labels_dir / filename, split)
        chosen = allocate_strata(rows, budget, args.seed)
        selected.extend(chosen)
        print(f"{split}: selected {len(chosen)} of {len(rows)} clips")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "DAiSEE_Compact_manifest.csv"
    columns = ("OfficialSplit", "SubjectID", "ClipID", *FIELDS)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in selected)
    (args.out_dir / "README.txt").write_text(
        "DAiSEE-Compact contains a deterministic manifest for 960 clips: 600 Train, 160 Validation, 200 Test.\n"
        "It retains the official subject-disjoint splits and samples observed four-label combinations.\n"
        "Videos are intentionally not included. Run this script with --copy-videos on an authorised local DAiSEE copy.\n",
        encoding="utf-8",
    )
    if args.copy_videos:
        copy_videos(selected, args.dataset_root, args.out_dir)
    print(f"Manifest written to {manifest}")


if __name__ == "__main__":
    main()
