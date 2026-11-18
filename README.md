# HOTC 2026 — HSI SAMURAI Tracking (Batch Evaluation & Submission)

A complete, reproducible single-object tracking pipeline for hyperspectral
(HSI) video. Raw hyperspectral cubes are converted to false-color RGB by a
learned band-selection module and tracked with a SAMURAI / SAM2.1 video
predictor. The pipeline evaluates whole sequence sets and writes a
competition submission CSV (with AUC / DP20 / IoU / Failure metrics when
ground truth is available).

---

## 1. What the pipeline does

1. **Load** hyperspectral frames for each sensor (`vis`, `nir`, `rednir`),
   de-mosaic to a spectral cube, sort bands, and resample onto a fixed
   wavelength grid.
2. **`FalseColorGen`** — a learned per-sensor spectral-band selector — maps each
   normalized cube to a 3-channel false-color image.
3. **`SamuraiTracker`** — a SAM2.1 (Hiera large) video predictor — is
   initialized from the first-frame box and propagates the target, returning an
   axis-aligned bounding box per frame.
4. **Evaluate & export** — runs over an entire split (`local` fold-0 or
   `official`/ranking), computes metrics when ground truth exists, and writes a
   submission CSV.

---

## 2. Package layout

```
submission_wc/
├── configs/
│   └── config.yaml               # All parameters (paths, sensors, tracker, run, output)
├── weights/
│   ├── sam2.1_hiera_large.pt     # SAMURAI/SAM2.1 backbone (large) — used by the pipeline
│   └── README_WEIGHTS.md         # Weight inventory + provenance
├── src/
│   ├── __init__.py
│   ├── track.py                  # BaseTracker + SamuraiTracker (SAM2.1)
│   └── val.py                    # FalseColorGen + HSI loading + eval/submission CLI
├── scripts/
│   ├── run_local_val.sh          # Local fold-0 evaluation (metrics)
│   └── run_submission.sh         # Official/ranking split → submission CSV
├── environment.yml               # Conda environment
├── requirements.txt              # pip dependencies
├── README.md                     # This file (overview)
└── RUNNING_INSTRUCTIONS.md       # Step-by-step execution guide
```

---

## 3. Key components

| Component | File | Role |
|-----------|------|------|
| `SamuraiTracker` | `src/track.py` | SAM2.1 video-predictor tracker (mask → bbox) |
| `FalseColorGen`  | `src/val.py`   | Learned HSI → false-color RGB band selection (per sensor) |
| HSI loading      | `src/val.py`   | De-mosaic (`x2cube`), band sort, grid resample, per-band normalize |
| Eval / CLI       | `src/val.py`   | Sequence discovery (local & official), metrics, submission CSV export |

---

## 4. Requirements (summary)

- Python 3.10+, PyTorch ≥ 2.3 (CUDA GPU strongly recommended)
- `numpy`, `opencv-python`, `tqdm`, `pandas` (required), `scikit-learn` (local mode)
- The **SAMURAI / SAM2** repository, installed locally (`pip install -e sam2`)
- The SAM2.1 backbone weight (see `weights/README_WEIGHTS.md`)

Full dependency lists: `requirements.txt` (pip) and `environment.yml` (conda).

---

## 5. Outputs

- **Submission CSV** (`official` mode): `ID,x,y,width,height` — aligned to
  `sample_submission.csv` when present, otherwise one row per predicted frame
  (`{sensor}-{name}_{frame:04d}`).
- **Console metrics** (when ground truth exists): per-sequence and aggregated
  **AUC**, **DP20**, **IoU**, **Failure** (overall and per sensor).

---

## 6. How to run

See **[RUNNING_INSTRUCTIONS.md](RUNNING_INSTRUCTIONS.md)** for the exact,
step-by-step commands (environment creation, dependency installation, weight
placement, path configuration, and execution).
