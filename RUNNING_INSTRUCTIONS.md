# Running Instructions — HOTC 2026 HSI SAMURAI Tracking (Batch / Submission)

Follow these steps in order. Commands assume you start from the
`submission_wc/` directory.

---

## 1. Create the environment

**Option A — venv:**
```bash
python3 -m venv venv
```

**Option B — conda:**
```bash
conda env create -f environment.yml
```

## 2. Activate the environment

```bash
# venv
source venv/bin/activate
# conda
conda activate hotc2026-wc
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```
(Install the PyTorch build matching your CUDA driver — see https://pytorch.org.)

## 4. Install SAMURAI / SAM2

```bash
git clone https://github.com/yangchris11/samurai.git
pip install -e samurai/sam2
```

## 5. Place model weights

Put the checkpoint in `weights/` (details in `weights/README_WEIGHTS.md`):

- `weights/sam2.1_hiera_large.pt` — SAM2.1 large backbone (used by `samurai_large`)

The SAM2 backbone must also be visible to the SAM2 repo at
`$SAMURAI_REPO_DIR/checkpoints/sam2.1_hiera_large.pt` (the tracker loads it from there).

## 6. Configure paths (environment variables)

The code reads these environment variables (the run scripts set them for you):

- `SAMURAI_REPO_DIR` — path to the local `sam2` repo root (read by `src/track.py`).
- `HOTC_BASE_DIR` — project root used by `src/val.py` to resolve dataset,
  training CSV, and sample-submission paths (defaults to `src/`).
- `HOTC_VAL_DIR` — validation / ranking split root (official mode).

`configs/config.yaml` documents every path and default in one place.

## 7. Run

**Official / ranking split → submission CSV (the actual submission):**
```bash
bash scripts/run_submission.sh "<path_to_official_or_ranking_split>" samurai_large
```
Equivalent direct call:
```bash
python src/val.py \
  --mode official \
  --tracker samurai_large \
  --val-dir <path_to_official_or_ranking_split> \
  --output results/submission_ranking2.csv \
  --workers 1 --decimals 6
```

**Local fold-0 evaluation (metrics; needs training data + scikit-learn):**
```bash
bash scripts/run_local_val.sh samurai_large
```

### CLI flags (`src/val.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--tracker`     | `samurai_base` (submission uses `samurai_large`) | `samurai_base` \| `samurai_large` |
| `--mode`        | `official` | `local` (fold-0 eval) \| `official` (submission CSV) |
| `--val-dir`     | `$HOTC_VAL_DIR` or ranking/validation | Split root |
| `--output`      | `<base>/submission_ranking2.csv` | Submission CSV path (official mode) |
| `--workers`     | `1` | Parallel sequences (ThreadPoolExecutor) |
| `--limit-seqs`  | `0` | Cap number of sequences (0 = all) |
| `--max-frames`  | `0` | Cap frames per sequence (0 = all) |
| `--decimals`    | `6` | Rounding for output boxes |

## 8. Check outputs

- **Submission CSV** at the `--output` path (default `results/submission_ranking2.csv`),
  columns `ID,x,y,width,height`.
- **Console report** with per-sequence and aggregated **AUC / DP20 / IoU /
  Failure** whenever ground truth is available for the split.
