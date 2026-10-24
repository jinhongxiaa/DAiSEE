# DAiSEE Compact Video Workflow

Source code for **A Compact Video Workflow for Four DAiSEE Affective States with Retrospective Evaluation**.

This repository implements the original 960-clip workflow: full-frame ResNet-18 embeddings, mean and temporal summaries, and four separate logistic-regression classifiers. It is not the later face-cropped MORSE/Q1 experiments and makes no state-of-the-art claim.

## What is included

- Deterministic local subset construction: 600 Train, 160 Validation, 200 Test clips.
- Six-frame full-image feature extraction: 512-dimensional mean embeddings and 6 × 512 sequences.
- Validation-only searches; a fixed historical final recipe for retrospective reproduction.
- Synthetic checks, environment capture, dataset instructions and provenance documentation.

**No dataset, labels, clip manifest, predictions, extracted features or pretrained weights are included.** Users obtain DAiSEE themselves. Exact historical software versions were not recovered; compatible installation constraints are supplied, not a historical lockfile.

## 1. Install

Use a separate Python **3.10 or 3.11** environment. These commands are intentionally single-line commands. Run them from the repository root. Do not paste terminal prompts or printed output into the shell.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In the commands below, Windows users can replace `python` with `.\.venv\Scripts\python.exe`; this avoids environment activation mistakes. Linux/macOS users should activate `.venv` first. CPU-only execution is supported; a GPU is not required. The tested scope for this packaging update is documented below; dependency installation and raw-video extraction were not rerun here.

## 2. Download DAiSEE and place it locally

Obtain authorized access from the [original DAiSEE provider](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) and follow its terms. The historical study downloaded [Kaggle mirror version 1](https://www.kaggle.com/datasets/olgaparfenova/daisee/download?datasetVersionNumber=1). Mirror/source equivalence is unverified. See [DATA_USAGE.md](DATA_USAGE.md).

Extract the download; move its `DataSet` and `Labels` directories into `data/DAiSEE/`. The required paths are:

| Local path from repository root | Content |
|---|---|
| `data/DAiSEE/Labels/TrainLabels.csv` | Original official Train labels |
| `data/DAiSEE/Labels/ValidationLabels.csv` | Original official Validation labels |
| `data/DAiSEE/Labels/TestLabels.csv` | Original official Test labels |
| `data/DAiSEE/DataSet/Train/<subject>/<clip-folder>/<clip>.avi` | Train videos; `.mp4` is also supported |
| `data/DAiSEE/DataSet/Validation/<subject>/<clip-folder>/<clip>.avi` | Validation videos |
| `data/DAiSEE/DataSet/Test/<subject>/<clip-folder>/<clip>.avi` | Test videos |

The extractors search recursively inside each split. Keep exact original clip filenames. If the release spells the directory `Dataset`, point `--video-root` to that actual directory. If its split names are `Training`/`Testing`, rename those directories to `Train`/`Test` for the extractors. Do not move subjects between splits. The original full download is approximately 15 GB; this package does not provide a separate compact video download.

## 3. Generate the compact manifest locally

```bash
python scripts/prepare_dataset.py
```

This creates `data/compact/DAiSEE_Compact_manifest.csv` using seed `20260928` and verifies its SHA-256 against the archived study manifest. It does **not** copy 960 videos or require an additional 1.6 GB copy; extractors use the existing full dataset. A mismatch stops this helper: do not claim exact replication or change the expected hash to conceal a source difference.

The subset builder reads all split labels to construct the retrospective stratified subset; this is distinct from fitting and selecting models using Train/Validation only. It is not a newly untouched Test protocol.

## 4. Check code, then extract features

```bash
python -m unittest discover -s tests -v
python run.py extract-mean
python run.py extract-temporal
```

First extraction downloads the standard ResNet-18 weights (about 45 MB) into the PyTorch cache. Internet is needed once. For interrupted HTTPS downloads, download the exact official file `https://download.pytorch.org/models/resnet18-f37072fd.pth` via a browser and put it in `<user-home>/.cache/torch/hub/checkpoints/` with that exact name; do not disable SSL verification. Both extractors use full frames, six uniformly spaced indices, torchvision preprocessing and inference mode. The legacy mean extractor and sequence extractor have different frame-decode failure rules: inspect errors and completeness, and do not silently replace their sampling logic.

`run.py` uses one numerical thread and batch size 4 to reduce CPU load. It does not promise a particular CPU percentage. Extraction has no per-clip resume facility; preserve completed archives. Nonempty output directories are protected from accidental reuse. For videos in another location:

```bash
python run.py extract-mean --video-root /path/to/DAiSEE/DataSet
python run.py extract-temporal --video-root /path/to/DAiSEE/DataSet
```

## 5. Search on Train/Validation

```bash
python run.py validate-mean
python run.py validate-temporal
```

Train: fit scalers/classifiers and PCA where applicable. Validation: search candidates. Test features/labels do not enter these search objectives. Mean C grid: `0.01,0.1,1,10,100`, class weighting balanced or none. Temporal C grid: `0.001,0.01,0.1,1,10`, balanced weights, PCA up to 128 whitened components. Temporal summary: mean, standard deviation, last-minus-first, mean absolute successive difference.

## 6. Optional retrospective final evaluation

```bash
python run.py evaluate
```

This command uses the **fixed historical recipe**, not an automatic choice from the newly generated validation tables: mean/C=10 for Boredom, mean/C=0.1 for Engagement and Confusion, temporal/C=1 for Frustration; balanced weights for all. It refits on Train+Validation and predicts Test. If a new environment produces different validation choices, record that difference; do not present this fixed recipe as newly selected.

Outputs are under `outputs/retrospective_final/`. At least two historical Test prediction sets existed; this is **not** a one-shot untouched Test. The legacy script's clip-stratified confidence intervals are not the revised manuscript participant-cluster intervals. See [provenance](docs/PROVENANCE.md). This package cannot regenerate the revision's fixed-prediction supplemental tables without the separately held historical predictions.

## 7. Record the actual environment

```bash
python scripts/capture_environment.py --output outputs/reproduction_environment.json
python -m pip freeze > outputs/requirements_resolved.txt
```

Record source dataset version, manifest hash, weight checksum, errors and hardware with each run. Never label the new environment as the unrecovered original one.

## Validation and limitations

Packaging checks: Python compilation; synthetic held-out-Test perturbation isolation for the mean search; deterministic subset selection; temporal-summary arithmetic; final fixed-recipe execution on synthetic aligned features. These are software checks, not new performance evidence. No raw-video run or exact historical-score reproduction was performed for this release. Python 3.11 installation commands are a recommended compatibility target, not a claim that a clean installation was executed here.

Historical reported macro-F1: Boredom **0.279**, Engagement **0.256**, Confusion **0.313**, Frustration **0.201**. Small subsets, rare-class performance and participant generalization limit interpretation. See [implementation specification](docs/IMPLEMENTATION_SPECIFICATION.md), which describes archived files separately held by the study, not bundled artifacts.

## GitHub, citation and DOI

Upload the contents of this directory to the repository root; keep this README visible at the top level. Do not upload generated `data/`, `outputs/`, `.venv/` or weights (manual web uploads may bypass `.gitignore`). See [Persian release instructions](docs/GITHUB_AND_DOI_FA.md).

Create a versioned GitHub Release and archive it with Zenodo, or deposit this code ZIP directly. No repository URL or DOI has been invented. Once issued, add the real DOI, authors and release information to this section and the manuscript Code Availability statement. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md): an authorized code license has not yet been supplied.
