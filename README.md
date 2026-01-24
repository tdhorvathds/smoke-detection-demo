---
title: Wildfire Smoke Detection Demo
emoji: 🔥
colorFrom: gray
colorTo: red
sdk: streamlit
python_version: "3.11"
app_file: app.py
pinned: false
---

# Wildfire Smoke Detection Demo  
**YOLOv11s vs Faster R-CNN | Streamlit**

An interactive **computer-vision demo** showcasing wildfire smoke detection with two production-grade object-detection architectures:

- **YOLOv11s** — fast, single-stage detector optimised for low-latency inference  
- **Faster R-CNN (ResNet-50 FPN v2)** — two-stage baseline with strong recall characteristics  

The application is designed as a **lightweight, browser-based demo** that allows non-technical users to explore model behaviour, robustness, and explainability without any local setup.

---

## Live demo
**Streamlit app:** 
Runs fully in the browser. No installation required for users.

---

## Why this project exists (industry perspective)
This demo focuses on **how models are packaged, deployed, and inspected**, not how they are trained.

It demonstrates:
- Turning a research-grade model into a **user-facing product**
- Clean separation between **model code, assets, and infrastructure**
- Practical handling of **large model artifacts** in a deployment-friendly way
- Making model behaviour **inspectable and explainable** for stakeholders

This mirrors real-world ML engineering workflows more closely than a training-only repository.

---

## App functionality

### Detect
Run live inference on:
- **Curated in-domain samples** with paired corruptions  
  - YOLO: *clean + noise*  
  - Faster R-CNN: *clean + fog*  
- **User-uploaded images** (PNG / JPG)

Key controls:
- Confidence threshold
- IoU threshold (YOLO only)
- Maximum number of detections
- Optional display of predicted and ground-truth boxes

Results are visualised directly on the image and summarised in a structured table.

---

### Explain
Explore **precomputed explainability overlays** (no runtime XAI computation):

- Methods: **Grad-CAM++**, **D-RISE**
- Variants: clean vs corrupted inputs (model-specific)

Features:
- Overlay-only or blended view
- Adjustable overlay opacity
- Optional predicted and GT boxes

This tab is intended for **qualitative inspection of attention patterns and failure modes**, not quantitative evaluation.

---

## Architecture & design choices

### 1) Two models, one interface
Both detectors expose the same inference contract:
```
image → bounding boxes + confidence scores
```

This allows:
- identical UI controls
- direct qualitative comparison
- architecture-agnostic visualisation code

---

### 2) Model loading & caching
- Models are loaded once per app instance using Streamlit’s resource caching
- No repeated weight loading on UI interaction
- Explicit CPU execution for cloud compatibility

This mirrors best practices for low-latency inference services.

---

### 3) Handling large model weights
- Weights are **not committed** to GitHub
- Stored as **GitHub Release assets**
- Downloaded automatically on first inference
- Cached locally for subsequent runs

This avoids Git LFS complexity while keeping the repository lightweight and reproducible.

---

### 4) Explainability as a first-class citizen
- XAI overlays are treated as **artifacts**, not ad-hoc outputs
- Decoupling XAI computation from the demo ensures:
  - fast UI response
  - deterministic visualisations
  - clean separation between research and presentation layers

---

## Repository structure

```
streamlit_app/
  app.py                 # Streamlit UI
  assets/
    samples_demo/        # Curated demo images (paired by corruption)
    xai_demo/            # Precomputed XAI overlays
    gt/                  # Optional COCO GT for sample images
  models/                # Empty in git (weights downloaded at runtime)
  src/
    rcnn_model.py        # Faster R-CNN model construction
  utils/
    yolo_inference.py
    rcnn_inference.py
    weights.py           # Auto-download from GitHub Releases

requirements.txt
README.md
```

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

Weights are downloaded automatically when inference is triggered.

---

## Limitations
- CPU-only execution (chosen for portability and cloud deployment)
- Explainability overlays are precomputed (no live XAI)
- Sample set is intentionally small and curated for demonstration purposes

---
