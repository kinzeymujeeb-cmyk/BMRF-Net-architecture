# BMRF-Net — Boundary-Aware Multi-Scale Referring Fusion

This repository contains code for BMRF-Net, a Boundary-Aware Multi-Scale
Referring Fusion network for referring camouflaged object detection (Ref-COD).
The model introduces a Feature Boundary Conditioning Module (FBCM), an
Attention-based Residual Max Pooling (ARMP) pathway, a Multi-Scale Referring
Feature Fusion (MRFF) module, and a Multi-Scale Feature Dissimilarity +
Sharpness decoder (MSFD) to improve boundary localisation and mask
completeness under reference guidance.

If you previously used this README as the LaTeX paper source, I can move
the full paper into `docs/PAPER.tex` or `PAPER.tex` — confirm if you want
that. The architecture figure (used in the paper) should be placed at
`figs/architecture_diagram.pdf` or `architecture_diagram.pdf` at repo root.

![BMRF-Net Architecture](figs/BMRF_Net%20Diagram.drawio.png)

This project reuses the R2C7K dataset and utilities; training, evaluation,
and inference scripts remain available in the repository.

## Key references
- Paper: Referring Camouflaged Object Detection (TPAMI 2025).

**Citation**

If you use this work, please cite:

```
@inproceedings{author2026bmrf,
  title={BMRF-Net: Boundary-Aware Multi-Scale Fusion for Referring Camouflaged Object Detection},
  author={Surname, Given and Surname, Given and ...},
  booktitle={Proceedings of ...},
  year={2026}
}
```

For questions or licensing inquiries, contact the authors (see original README contacts).

## Project structure
Top-level files and directories:

| Path | Purpose |
| --- | --- |
| `train.py` | Training entrypoint. Edit `data_root` and hyperparams as needed. |
| `test.py` | Evaluation script (computes metrics and saves predictions). |
| `infer.py` | Inference pipeline for generating foreground / mask maps. |
| `make_comparison.py` | Utilities to compare predictions across methods / checkpoints. |
| `models/` | Model definitions and backbone implementations (includes `pvt_v2.py`, `bmrf_net.py`, `modules.py`). The implemented backbone is PVT-v2-B2 and the repository contains the BMRF-Net modules: FBCM, ARMP, MRFF, and MSFD. |
| `data/` | Dataset loaders and helpers (`refdataset.py`, `utils.py`). |
| `utils/` | Metrics and utility scripts used across training/testing. |
| `snapshot/` | Place checkpoints here (`snapshot/saved_models/`). |
| `figs/` | Figures used in the README and paper (e.g., `architecture.png`). |

## Dataset: R2C7K
R2C7K contains two subsets: `Camo` (camouflaged images) and `Ref` (reference images / features). Required structure:

```
R2C7K/
  ├─ Camo/
  │   ├─ train/        # training images and masks (64 categories)
  │   └─ test/         # testing images and masks (64 categories)
  └─ Ref/
      ├─ Images/        # reference images (64 categories)
      ├─ RefFeat_ICON-R/ # precomputed reference object representations
      └─ Saliency_ICON-R/ # foreground maps used to generate object masks
```

Set `data_root` in `train.py`, `test.py`, and `infer.py` to point to your `R2C7K` folder.

## Environment setup
Preferred (Conda):

```bash
conda env create -f environment.yml
conda activate refcod
```

If you prefer pip, create `requirements.txt` from the Conda env and then:

```bash
pip install -r requirements.txt
```

Note: `environment.yml` is included in the repository. Adjust CUDA / PyTorch versions to match your GPU.

## Quick start: training, evaluation, inference

1) Train (example):

```bash
python train.py --data_root /path/to/R2C7K --batch_size 16 --lr 5e-4 --backbone_lr 5e-5 --epochs 60
```

2) Evaluate (test):

```bash
# Ensure checkpoint saved in snapshot/saved_models/
python test.py --checkpoint snapshot/saved_models/bmrf_net.pth --data_root /path/to/R2C7K
```

3) Inference (single image or folder):

```bash
python infer.py --checkpoint snapshot/saved_models/bmrf_net.pth --input /path/to/images --output outputs/
```

4) Make comparisons between methods:

```bash
python make_comparison.py --pred_dir outputs/ --gt_dir /path/to/R2C7K/Camo/test
```

## Evaluation metrics
Supported metrics (computed in `utils/metrics.py`):
- Structure-measure (`S_m` / Sm)
- Adaptive E-measure (`adpE`)
- Weighted F-measure (`wF`)
- Mean Absolute Error (`MAE` / `M`)

Use `test.py` to compute these metrics on test splits and save per-image predictions to `preds/`.

## Checkpoints
- Place model checkpoints under `snapshot/saved_models/` (e.g., `bmrf_net.pth`).

## Notes & acknowledgements
This repo builds on prior COD frameworks including SINet-V2, PFENet and MethodsCmp. See original README for dataset download links and additional figures. The original author contacts and dataset links remain valid.

If you'd like, I can also:
- Extract a `requirements.txt` from `environment.yml`.
- Add example config files and a short script to run end-to-end experiments.

---


# BMRF-Net-architecture
