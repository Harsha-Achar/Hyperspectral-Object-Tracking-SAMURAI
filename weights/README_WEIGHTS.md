# Model Weights

This folder contains the weight used by the submission pipeline
(tracker = `samurai_large`, the configuration used to produce
`submission_ranking2.csv`).

| File | Size | Purpose | Status in this package |
|------|------|---------|------------------------|
| `sam2.1_hiera_large.pt` | ~857 MB | SAMURAI / SAM2.1 backbone used by `--tracker samurai_large` | ✅ included |

## FalseColorGen

No external FalseColorGen checkpoint is required or shipped. `FalseColorGen`
runs with its built-in (default) per-sensor band-selection initialization
(`src/val.py`).

## SAM2 backbone location at runtime

`src/track.py` loads the SAM2 backbone from the SAMURAI repository, i.e.
`$SAMURAI_REPO_DIR/checkpoints/sam2.1_hiera_large.pt` — **not** from this
folder. The copy here is the bundled submission artifact; make sure the same
file is also present in the SAMURAI repo's `checkpoints/` directory (or point
`SAMURAI_REPO_DIR` at a checkout that already contains it).

## Model weights drive link

link : https://drive.google.com/drive/folders/1MMHwgxff4WZpT8d97MLFqWsmSn0G5vAu?usp=sharing