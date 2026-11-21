# VisionGuard — Competition Delivery & Live Demonstration Guide

## 1. Project Abstract
VisionGuard is an end-to-end industrial visual anomaly detection and defect localization platform that learns strictly from defect-free product images to detect and localize unseen manufacturing defects at the pixel level. It systematically implements and benchmarks reconstruction-based (Convolutional Autoencoder) and feature-based (PaDiM, PatchCore, FastFlow) architectures under identical, zero-leakage evaluation protocols across MVTec AD, VisA, MVTec LOCO AD, and MVTec AD 2, alongside controlled studies on normal training-data efficiency (10%–100%), six industrial image degradations, and PatchCore coreset/layer ablations.

## 2. Five-Minute Live Demonstration Walkthrough
1. **0:00–0:45 (Executive Overview)**: Present the primary research question and highlight scientific honesty — metrics display `"Not evaluated"` until models are actually fitted and evaluated.
2. **0:45–2:00 (Model Laboratory)**: Train and compare Autoencoder, PaDiM, PatchCore, and FastFlow live on MVTec AD `bottle`. Walk through Image AUROC, Pixel AUROC, AUPRO, F1/Precision/Recall at the normal-validation threshold, ROC/PR curves, Confusion Matrix, and Checkpoint Size.
3. **2:00–3:15 (Live Inspection)**: Switch to Live Inspection, run real inference on unseen test samples or an uploaded PNG/JPG image, toggle overlay intensity, compare Autoencoder reconstruction vs PatchCore localization, show the decoupled Image Quality Assessment panel, and export a structured JSON inspection report.
4. **3:15–4:30 (Research Experiments)**: Run Data Efficiency (10%–100%), Robustness Degradation (Gaussian noise, blur, brightness, contrast, JPEG compression, resize/resample), and Cross-Dataset / LOCO AD structural vs logical difficulty analysis.
5. **4:30–5:00 (Reproducibility & Architecture)**: Export machine-readable JSON/CSV evidence and showcase the accompanying Python 3.11 / PyTorch WideResNet-50-2 / pytest / Streamlit research repository.

## 3. Core Technical Q&A for Judges
- **Why normal-only learning?** Defective samples are rare and novel defect types appear continuously in manufacturing; supervised binary classifiers cannot generalize to unseen defect modes.
- **Why are anomaly scores not probabilities?** Raw outputs are unbounded reconstruction errors, Mahalanobis distances, or k-NN distances; thresholds are calibrated solely on held-out normal validation images (`μ + 2.5σ`) without test-label leakage.
- **Why do patch detectors struggle on LOCO AD logical anomalies?** Local patches appear normal individually even when global count, placement, or presence constraints are violated.
- **Why keep Image Quality Assessment separate?** Optical blur or poor factory illumination indicates a camera/acquisition issue rather than a physical part defect.
