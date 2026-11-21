# VisionGuard — Complete User Guide & Judge Showcase Playbook

This guide explains how to use **VisionGuard**, how to explain it simply to non-technical audiences, and how to run a high-impact live demonstration for competition judges.

---

## 1. How to Explain VisionGuard Simply (30-Second Elevator Pitch)

### For Non-Technical People / Friends / General Audience
> *"Imagine a factory making thousands of glass bottles, metal nuts, or circuit boards every hour. Usually, almost all products are good, and defects are rare and unpredictable — a crack today, a stain tomorrow, a missing wire next week. Normal AI needs thousands of broken examples to learn, which factories don't have.*
>
> *VisionGuard flips the problem: we only show the AI **good, defect-free products**. Once it learns what 'normal' looks like, it can inspect any brand-new product, flag if anything looks wrong, and paint a glowing heatmap directly over the exact scratch, crack, or stain."*

### One-Line Technical Summary (For Judges & Professors)
> *"VisionGuard is an end-to-end unsupervised industrial anomaly detection and pixel-level localization platform that systematically compares reconstruction-based (Autoencoder) and feature-based (PaDiM, PatchCore, FastFlow) models across MVTec AD, VisA, MVTec LOCO AD, and MVTec AD 2 under strict zero-leakage scientific protocols."*

---

## 2. Step-by-Step: How to Use VisionGuard

### Step 1 — Start at **Executive Overview**
- When you first open VisionGuard, notice that metrics say **"Not evaluated"** instead of showing fake numbers.
- Click **"Train & Evaluate 4 Models"** (or **"Run Core Benchmark"** in the top bar).
- In a few seconds, VisionGuard trains real **Autoencoder**, **PaDiM**, **PatchCore**, and **FastFlow** models on normal-only `bottle` images, calibrates thresholds on held-out normal validation images, and populates live **Image AUROC**, **Pixel AUROC**, and **F1** charts.

### Step 2 — Explore & Train in **Model Laboratory**
1. Look at the **Model Engine Maturity Catalog** at the top:
   - **PatchCore (`Optimized`)** — Fast coreset memory-bank nearest-neighbor engine; typically strongest overall recall and speed balance.
   - **PaDiM (`Stable`)** — Per-patch Gaussian distribution with Mahalanobis distance.
   - **Convolutional Autoencoder (`Stable`)** — Reconstruction baseline; produces a visual reconstructed image.
   - **FastFlow (`Experimental`)** — 2D invertible normalizing-flow density estimator.
2. Pick any **Dataset** (`MVTec AD`, `VisA`, `MVTec LOCO AD`, `MVTec AD 2`), **Category** (e.g., `bottle`, `metal_nut`, `pcb1`, `breakfast_box`), **Resolution**, and **Seed**.
3. Click **"Train & Evaluate"** for one model, or **"Compare All 4 Models"** for a side-by-side benchmark.
4. Click any row in the **Verified Benchmark Comparison Table** to inspect its **ROC Curve**, **Precision-Recall Curve**, **Confusion Matrix**, and **Training Loss Curve**.
5. Use the **Maturity Filter** (`All Tiers | Optimized | Stable | Experimental`) or **Dataset Filter**, and click **Export CSV** or download any model's **JSON checkpoint**.

### Step 3 — Run Real-Time Defect Detection in **Live Inspection**
1. Select a **Dataset**, **Category**, and **Model Architecture** (if that model hasn't been trained yet, click **"Train Model First"**).
2. Either:
   - Click any unseen test thumbnail on the left (`OK` = normal, `DEF` = defective), **OR**
   - Click **"Upload PNG / JPG"** to inspect your own photo from disk.
3. Click **"Run Live Inspection"**.
4. Review the 4 visual panels:
   - **1. Input** — Preprocessed inspection image.
   - **2. Anomaly Heatmap** — Brighter yellow/orange regions mark higher anomaly scores.
   - **3. Localization Overlay** — Heatmap blended directly onto the product (adjust with the **Overlay Intensity** slider).
   - **4. AE Reconstruction / Ground-Truth Mask** — Shows what the Autoencoder tried to reconstruct, or the true defect mask for test samples.
5. Check the **Image Quality Assessment** box: it flags blur, bad lighting, low contrast, low resolution, or heavy JPEG compression *separately* so a blurry camera shot is never confused with a broken part.
6. Click **"Download Inspection Report"** to save a structured JSON report.

### Step 4 — Inspect & Import Real Datasets in **Dataset Explorer**
- Switch tabs across **MVTec AD**, **VisA**, **MVTec LOCO AD**, **MVTec AD 2**, and **BTAD** to inspect official licenses, citations, folder structures, and sample/mask galleries.
- **Using Official Real Datasets**: After downloading an official dataset (e.g., MVTec AD `bottle` folder) onto your laptop, click **"Import Real Local Folder"** and select that category folder. VisionGuard validates image readability, mask alignment, and split isolation automatically and switches the provenance label to **Local Real Data (User Verified)**.

### Step 5 — Run Controlled Studies in **Research Experiments**
- **Exp B: Data Efficiency** — Tests 10%, 25%, 50%, and 100% normal training images to show which models stay accurate when factory data is scarce.
- **Exp C: Robustness** — Tests 6 real-world factory camera degradations (Gaussian noise, blur, brightness drift, contrast drop, JPEG compression, resize/resampling) at severities 1, 3, and 5 without corrupting ground-truth masks.
- **Exp D: PatchCore Ablations** — Tests Coreset Ratio (10%–100%), Feature Layers (`L1`, `L2`, `L1+L2`, `L1+L2+L3`), and Resolution (`64px` vs `96px`) to prove the speed vs accuracy trade-off.
- **Exp E/F: Cross-Dataset & Category Difficulty** — Benchmarks 8 categories across all 4 datasets independently and highlights why **MVTec LOCO AD logical anomalies** (e.g., missing orange or extra screw) are harder for local patch models than surface scratches.

### Step 6 — Audit & Export in **Inspection History**
- View every live inspection's timestamp, model, category, filename, Normal/Anomalous decision, raw score vs threshold, quality warnings, and measured latency.
- Search, filter by Normal/Anomalous, export CSV, or clear history.

---

## 3. Winning 5-Minute Live Demo Flow for Competition Judges

Follow this exact sequence during your presentation so every click tells a scientific story:

| Time | Page | What to Click | What to Say to Judges |
| :--- | :--- | :--- | :--- |
| **0:00–0:45** | **Executive Overview** | Point out `"Not evaluated"` before clicking anything | *"First, notice there are no hardcoded fake numbers. Every metric in VisionGuard is computed live from real training and unseen test evaluation."* |
| **0:45–1:45** | **Model Laboratory** | Click **Compare All 4 Models** on `MVTec AD / bottle`, then click the **PatchCore** row | *"We compare four distinct paradigms — Autoencoder, PaDiM, PatchCore, and FastFlow — tagged by engineering maturity. Notice thresholds are calibrated strictly on held-out normal validation images, never on test labels."* Point to the live ROC curve, Pixel AUROC, AUPRO, and Confusion Matrix. |
| **1:45–3:00** | **Live Inspection** | Pick a defective (`DEF`) sample → **Run Live Inspection** → adjust **Overlay Intensity** → switch to **Autoencoder** to show reconstruction → click **Download Inspection Report** | *"Here is real-time pixel localization. The glowing overlay isolates the exact defect, while the separate Image Quality Assessment ensures a blurry or dark camera feed raises a reliability caution rather than a false part failure."* |
| **3:00–4:15** | **Research Experiments** | Open **Exp B (Data Efficiency)** or **Exp C (Robustness)** → click Run; or show **Exp E/F** on `MVTec LOCO AD` | *"Beyond building an app, we answer a clear research question: how do these models behave with only 10% normal data, under factory noise/blur, and on logical anomalies where every patch looks normal individually?"* |
| **4:15–5:00** | **Dataset Explorer + Methodology** | Show **Import Real Local Folder** and **Python 3.11 / PyTorch CLI** tab | *"Verification fixtures let us demo offline safely, and any official MVTec AD, VisA, LOCO AD, or AD 2 folder can be imported directly—plus our full PyTorch WideResNet-50-2, Streamlit, and pytest codebase is packaged for GPU/CPU benchmarking."* |

---

## 4. Real-World Use Cases You Can Mention to Judges

1. **Electronics & PCB Inspection (VisA `pcb1`)**: Catching missing SMD resistors, solder bridges, or trace scratches on circuit boards without labeling every possible soldering error.
2. **Automotive & Precision Machining (MVTec AD `metal_nut`, `screw`, `sheet_metal`)**: Detecting micro-cracks, surface dents, and machining burrs on brushed metal components.
3. **Pharmaceutical & Packaging Quality (`capsule`, `bottle`)**: Spotting cracked glass rims, contaminated capsules, or missing labels on high-speed conveyor belts.
4. **Assembly Completeness Verification (`MVTec LOCO AD`)**: Highlighting where local vision detectors need to be paired with global relational checks (counting screws/washers in hardware kits).

---

## 5. Top 5 Tough Judge Questions — Short, Confident Answers

1. **"Are these real ML models or pre-saved results?"**
   - *"100% real. Before you click Train, every score reads 'Not evaluated'. When we click Train, the Autoencoder runs real Adam backpropagation, PaDiM inverts per-patch covariance matrices for Mahalanobis distance, PatchCore runs greedy k-center coreset sampling and k-NN search, and FastFlow fits invertible affine coupling layers live."*
2. **"Why didn't you just train a normal defect classifier (CNN/YOLO)?"**
   - *"In factories, defects are rare and new defect types appear unexpectedly. Supervised classifiers fail on unseen defect types. Normal-only anomaly detection only needs healthy parts and flags any deviation."*
3. **"Is that anomaly score a probability like 85%?"**
   - *"No — we deliberately never call it a probability. It's a continuous distance or reconstruction error magnitude, and the Normal/Anomalous cutoff comes strictly from the held-out normal validation distribution (μ + 2.5σ) with zero test-label leakage."*
4. **"What happens if the factory camera goes out of focus or dims?"**
   - *"We built a decoupled Image Quality Assessment module (Laplacian blur variance, luminance, RMS contrast, JPEG blockiness) that warns the operator about camera quality separately, and we quantified exact AUROC drop across 6 degradations in Experiment C."*
5. **"What is your biggest limitation?"**
   - *"Local patch-based methods like PatchCore and PaDiM excel at surface/structural defects, but struggle on MVTec LOCO AD logical anomalies (like a missing component in a kit) where every individual patch looks normal. Identifying and proving that limitation experimentally is part of our research contribution."*

---

## 6. Pushing Your Project to GitHub

From your local computer (where your GitHub credentials or Personal Access Token are configured):
```bash
git push -u origin main
```
Note: `.gitignore` is already configured so raw licensed third-party datasets (`data/raw/`) and large checkpoints are never accidentally pushed to GitHub.
