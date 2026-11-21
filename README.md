# VisionGuard
### Intelligent Industrial Defect Detection, Anomaly Localization, and Visual Inspection Platform

VisionGuard is a dual-runtime industrial visual anomaly detection platform built for reproducible research, controlled ML benchmarking, and live competition demonstration:

1. **Interactive Web Platform (`src/`, React + TypeScript + Tailwind + Recharts)**:
   - Implements real, mathematically authentic **Convolutional Autoencoder**, **PaDiM** (multivariate Gaussian + Mahalanobis), **PatchCore** (locally-aware memory bank + greedy k-center coreset + k-NN), and **FastFlow** (2D invertible affine coupling normalizing flow) running live on pixel tensors.
   - Strict scientific validity: normal-only training, held-out normal validation threshold calibration (zero test-label leakage), real Image AUROC, Pixel AUROC, AUPRO (8-connected components), F1/Precision/Recall, ROC/PR curves, Confusion Matrix, and measured inference latency.
   - Independent **Image Quality Assessment** (blur, illumination, contrast, resolution, JPEG blockiness) decoupled from defect predictions.
   - All 7 enterprise dashboard views: Executive Overview, Live Inspection, Model Laboratory, Dataset Explorer (with local real dataset folder import for MVTec AD / VisA / LOCO AD / AD 2), Research Experiments (Experiments B–F), Inspection History, and Methodology & Competition Kit.

2. **Python 3.11 Research & Production Suite (`python/`)**:
   - PyTorch Convolutional Autoencoder, WideResNet-50-2 PaDiM, WideResNet-50-2 PatchCore, unified multi-dataset adapters (`MVTec AD`, `VisA`, `MVTec LOCO AD`, `MVTec AD 2`, optional `BTAD`), dataset validation CLI, corruption robustness suite, pytest unit tests, and Streamlit companion dashboard.

---

## Primary Research Question

> *"How effectively do reconstruction-based and pretrained feature-based anomaly detection methods identify and localize unseen industrial defects across different product categories and datasets, and how does their performance change under limited normal training data and realistic image-quality degradation?"*

---

## Quick Start

### A. Live Interactive Web Platform (Node.js + Vite)
```bash
npm install
npm run dev
# Opens on http://localhost:3000
```

### B. Python 3.11 + PyTorch + Streamlit Research Platform
```bash
cd python
python -m venv .venv
source .venv/bin/activate   # Windows: .\scripts\start_windows.ps1 -InstallDeps
pip install -r requirements.txt

# Validate local dataset folder
python -m src.data.validation --dataset mvtec_ad --data-dir data/raw/mvtec_ad

# Train and evaluate PatchCore / PaDiM / Autoencoder
python -m src.training.train --model patchcore --dataset mvtec_ad --category bottle --seed 42
python -m src.training.train --model padim --dataset mvtec_ad --category bottle --seed 42
python -m src.training.train --model autoencoder --dataset mvtec_ad --category bottle --seed 42

# Run unit tests & launch Streamlit dashboard
pytest -q
streamlit run dashboard/app.py
```

---

## Dataset Provenance & Licensing

Official third-party datasets are **never redistributed** in this repository:
- **MVTec AD** (`CC BY-NC-SA 4.0`): https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- **VisA** (`CC BY 4.0`): https://registry.opendata.aws/visa/
- **MVTec LOCO AD** (`CC BY-NC-SA 4.0`): https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad
- **MVTec AD 2** (`CC BY-NC-SA 4.0`, public evaluation split only): https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2
- **BTAD** (optional): http://avires.dimi.uniud.it/papers/btad/btad.zip

See `docs/DATASETS.md`, `docs/ARCHITECTURE.md`, and `docs/COMPETITION_DEMO.md` for full acquisition instructions, methodology, and presentation guides.
