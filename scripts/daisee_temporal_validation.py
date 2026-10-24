#!/usr/bin/env python3
r"""Validation-only temporal-profile selection for DAiSEE Compact.

The profile concatenates frame mean, standard deviation, end-to-start change,
and mean absolute frame-to-frame change. Model selection uses Train/Validation
only; the Test labels are never used.
"""
from __future__ import annotations
import argparse, csv, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TASKS = ("Boredom", "Engagement", "Confusion", "Frustration")

def parse_grid(s: str):
    try: values = sorted({float(x) for x in s.split(",")})
    except ValueError as exc: raise argparse.ArgumentTypeError("--c-grid must be numbers") from exc
    if not values or any(x <= 0 for x in values): raise argparse.ArgumentTypeError("--c-grid values must be positive")
    return values

def profile(x: np.ndarray) -> np.ndarray:
    return np.concatenate((x.mean(1), x.std(1), x[:, -1] - x[:, 0], np.abs(np.diff(x, axis=1)).mean(1)), axis=1)

def measure(y, p):
    precision, recall, _, _ = precision_recall_fscore_support(y, p, average="macro", labels=[0,1,2,3], zero_division=0)
    return {"macro_f1": f1_score(y, p, average="macro", labels=[0,1,2,3], zero_division=0),
            "weighted_f1": f1_score(y, p, average="weighted", labels=[0,1,2,3], zero_division=0),
            "accuracy": accuracy_score(y,p), "macro_precision": precision, "macro_recall": recall}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--features",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--c-grid",type=parse_grid,default=parse_grid("0.001,0.01,0.1,1,10")); p.add_argument("--pca-components",type=int,default=128); p.add_argument("--seed",type=int,default=20260928); a=p.parse_args()
    data=np.load(a.features,allow_pickle=False)
    if data["X"].ndim != 3: raise ValueError("Expected [clips, frames, embedding_dim] sequence embeddings")
    tr, va = data["split"]=="Train", data["split"]=="Validation"
    if not tr.any() or not va.any(): raise ValueError("Missing official Train or Validation split")
    X=profile(data["X"]); components=min(a.pca_components, int(tr.sum())-1, X.shape[1]); a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for task in TASKS:
        for c in a.c_grid:
            model=Pipeline([("scale",StandardScaler()),("pca",PCA(n_components=components,whiten=True,random_state=a.seed)),("classifier",LogisticRegression(C=c,class_weight="balanced",max_iter=5000,random_state=a.seed))])
            model.fit(X[tr],data[task][tr]); rows.append({"task":task,"C":c,"pca_components":components,**measure(data[task][va],model.predict(X[va]))})
    chosen=[]
    for task in TASKS: chosen.append(max((x for x in rows if x["task"]==task),key=lambda x:(x["macro_f1"],x["weighted_f1"])))
    for name,content in (("temporal_validation_results.csv",rows),("temporal_selected_candidates.csv",chosen)):
        with (a.output/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(content[0])); w.writeheader(); w.writerows(content)
    (a.output/"temporal_validation_protocol.json").write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),"test_labels_used":False,"feature_profile":"mean+standard-deviation+end-to-start-change+mean-absolute-change","selection_metric":"macro_f1","pca_components":components,"c_grid":a.c_grid},indent=2),encoding="utf-8")

if __name__=="__main__": main()
