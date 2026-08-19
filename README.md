# BMRF-Net: Boundary-Aware Multi-Scale Fusion for Referring Camouflaged Object Detection

Official PyTorch implementation of **BMRF-Net** (Boundary-Aware Multi-Scale Referring Fusion Network) for Referring Camouflaged Object Detection (Ref-COD).

BMRF-Net introduces explicit boundary reasoning before reference interaction and preserves structural details throughout multi-scale decoding:
- **Feature Boundary Conditioning Module (FBCM):** Estimates an explicitly supervised boundary prior ($F_e$) from hierarchical query features ($f_2, f_4$) using gated multi-dilation convolutions.
- **Attention-based Residual Max Pooling (ARMP):** Enhances retained query features ($f_2, f_3, f_4$) using the predicted boundary prior before reference interaction.
- **Multi-Scale Referring Feature Fusion (MRFF):** Refines target representations coarse-to-fine across pyramid bin sizes ($11\times11, 22\times22, 44\times44$) guided by reference cross-correlation matching ($M$).
- **MSFD Decoder:** Combines multi-scale feature dissimilarity with a sharpness head to reconstruct complete masks with sharp contours.

![BMRF-Net Architecture](figs/BMRF_Net%20Diagram.drawio.png)

---

## 📊 Benchmark Results

### 1. Comparison on R2C7K Benchmark (Overall, Single-Object, and Multi-Object)
BMRF-Net outperforms baseline methods and existing Ref-COD models, achieving a **30.56% relative error reduction in MAE** over the baseline R2CNet.

| Method | Venue | Overall $S_m \uparrow$ | Overall $adpE \uparrow$ | Overall $wF \uparrow$ | Overall $M \downarrow$ | Single $S_m \uparrow$ | Single $adpE \uparrow$ | Single $wF \uparrow$ | Single $M \downarrow$ | Multi $S_m \uparrow$ | Multi $adpE \uparrow$ | Multi $wF \uparrow$ | Multi $M \downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SINet-V2-Ref | TPAMI'22 | 0.823 | 0.888 | 0.700 | 0.033 | 0.828 | 0.889 | 0.705 | 0.032 | 0.771 | 0.874 | 0.634 | 0.043 |
| BSA-Net-Ref | AAAI'22 | 0.830 | 0.912 | 0.727 | 0.030 | 0.827 | 0.913 | 0.733 | 0.030 | 0.774 | 0.895 | 0.655 | 0.039 |
| BGNet-Ref | IJCAI'22 | 0.840 | 0.909 | 0.738 | 0.029 | 0.844 | 0.910 | 0.742 | 0.029 | 0.792 | 0.887 | 0.679 | 0.036 |
| DGNet-Ref | MIR'23 | 0.821 | 0.891 | 0.696 | 0.032 | 0.827 | 0.890 | 0.703 | 0.031 | 0.748 | 0.879 | 0.607 | 0.045 |
| FSEL-Ref | ECCV'24 | 0.851 | 0.906 | 0.751 | 0.027 | 0.856 | 0.907 | 0.756 | 0.026 | 0.798 | 0.890 | 0.688 | 0.036 |
| GLCONet-Ref | TNNLS'25 | 0.850 | 0.905 | 0.747 | 0.027 | 0.859 | 0.913 | 0.754 | 0.025 | 0.799 | 0.908 | 0.700 | 0.032 |
| R2CNet (Baseline) | TPAMI'25 | 0.805 | 0.879 | 0.669 | 0.036 | 0.810 | 0.880 | 0.674 | 0.035 | 0.747 | 0.872 | 0.602 | 0.046 |
| **BMRF-Net (Ours)** | - | **0.854** | **0.917** | **0.757** | **0.025** | **0.871** | **0.916** | **0.782** | **0.021** | **0.843** | **0.917** | **0.741** | **0.028** |

---

### 2. Generalization Performance on Standard COD Benchmarks

| Methods | CAMO $S_m \uparrow$ | CAMO $adpE \uparrow$ | CAMO $wF \uparrow$ | CAMO $M \downarrow$ | CHAMELEON $S_m \uparrow$ | CHAMELEON $adpE \uparrow$ | CHAMELEON $wF \uparrow$ | CHAMELEON $M \downarrow$ | COD10K $S_m \uparrow$ | COD10K $adpE \uparrow$ | COD10K $wF \uparrow$ | COD10K $M \downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SINet | **0.751** | 0.606 | 0.561 | 0.113 | 0.869 | 0.740 | 0.771 | 0.044 | 0.771 | 0.551 | 0.506 | 0.080 |
| PiCANet | 0.701 | 0.573 | 0.716 | 0.125 | 0.765 | 0.618 | 0.779 | 0.085 | 0.696 | 0.489 | 0.712 | 0.081 |
| BASNet | 0.615 | 0.503 | 0.671 | 0.124 | 0.847 | 0.795 | **0.883** | 0.044 | 0.661 | 0.486 | 0.729 | 0.071 |
| EGNet | 0.737 | 0.655 | **0.758** | 0.102 | 0.856 | 0.766 | **0.883** | 0.049 | 0.751 | 0.595 | **0.793** | 0.053 |
| **BMRF-Net (Ours)** | 0.742 | **0.805** | 0.636 | **0.094** | **0.880** | **0.945** | 0.816 | **0.030** | **0.838** | **0.919** | 0.737 | **0.028** |

---

### 3. Ablation Study of Proposed Modules on R2C7K Test Set

| ID | FBCM | MRFF | ARMP | $S_m \uparrow$ | $adpE \uparrow$ | $wF \uparrow$ | $M \downarrow$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (a) | | | | 0.805 | 0.879 | 0.669 | 0.036 |
| (b) | ✓ | | | 0.836 | 0.902 | 0.722 | 0.030 |
| (c) | | ✓ | | 0.833 | 0.899 | 0.716 | 0.031 |
| (d) | | | ✓ | 0.829 | 0.895 | 0.708 | 0.032 |
| (e) | ✓ | ✓ | | 0.848 | 0.911 | 0.745 | 0.027 |
| (f) | ✓ | | ✓ | 0.846 | 0.909 | 0.741 | 0.028 |
| (g) | | ✓ | ✓ | 0.844 | 0.907 | 0.737 | 0.028 |
| (h) | **✓** | **✓** | **✓** | **0.854** | **0.917** | **0.757** | **0.025** |

---

## 📁 Repository Structure

```text
├── data/                  # Dataset loaders and augmentations (refdataset.py)
├── figs/                  # Architectural diagrams and qualitative visuals
├── models/                # Model architecture definitions
│   ├── bmrf_net.py        # Core BMRF-Net framework
│   ├── modules.py         # FBCM, ARMP, MRFF, and MSFD modules
│   └── pvt_v2.py          # PVT-v2-B2 transformer backbone
├── snapshot/              # Directory for model checkpoints
├── utils/                 # Metrics (Sm, adpE, wF, MAE) and loss functions
├── infer.py               # Single image / batch inference script
├── make_comparison.py     # Qualitative visual comparison tool
├── test.py                # Benchmark evaluation script
└── train.py               # Training entrypoint
Dataset: R2C7K
Organize the R2C7K dataset under your data directory as follows:
R2C7K/
  ├── Camo/
  │   ├── train/          # Training images and masks (64 categories)
  │   └── test/           # Testing images and masks (64 categories)
  └── Ref/
      ├── Images/         # Reference images (64 categories)
      ├── RefFeat_ICON-R/ # Precomputed reference object representations (ICON-R)
      └── Saliency_ICON-R/# Foreground saliency maps
Set data_root in train.py, test.py, and infer.py to point to your R2C7K directory.
Environment Setup
# Using Conda
conda env create -f environment.yml
conda activate refcod

# Or using pip
pip install -r requirements.txt
Quick Start1.
1. Training
Train
python train.py --data_root /path/to/R2C7K --batch_size 16 --lr 5e-4 --backbone_lr 5e-5 --epochs 60
2. Evaluation
Evaluate the trained model under the standard 5-shot reference setting:
python test.py --checkpoint snapshot/saved_models/bmrf_net_best.pth --data_root /path/to/R2C7K
3. Inference
Run inference for a single image or folder:
python infer.py --checkpoint snapshot/saved_models/bmrf_net_best.pth --input /path/to/images --output outputs/
4. Qualitative Comparisons
python make_comparison.py --pred_dir outputs/ --gt_dir /path/to/R2C7K/Camo/test
Evaluation Metrics
Evaluated via utils/metrics.py:
Structure-measure (Sm)
Adaptive E-measure (adpE)
Weighted F-measure (wF)
Mean Absolute Error (MAE)
