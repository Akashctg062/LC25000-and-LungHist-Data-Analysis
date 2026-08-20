# LC25000 Histopathology Classifier — Web App

An interactive web application that classifies lung and colon histopathology
images using **all 12 deep learning models** from the LC25000 comparison study.

## Features

- **Single Image** — upload one image, get predictions from all 12 models
  side-by-side, a consensus vote, a probability matrix, and Grad-CAM heatmaps.
- **Batch Analysis** — upload many images at once, get a results table and a
  downloadable CSV.
- **History** — every prediction made during the session, exportable as CSV.
- **Grad-CAM** — visual heatmaps showing which image regions drove each
  prediction (available for the 9 convolutional models).
- Runs on **GPU if available, CPU otherwise** — so it works locally on your
  RTX 5060 now and can be deployed to a CPU-only cloud server later.

---

## File structure

```
webapp/
├── app.py            ← the Streamlit application (run this)
├── models.py         ← all 12 model architectures + loading logic
├── gradcam.py        ← Grad-CAM heatmap implementation
├── requirements.txt  ← Python dependencies
└── README.md         ← this file
```

You also need your **trained model checkpoints** — the 12 `.pth` files produced
by the training notebook. By default the app looks in:

```
E:\LC25000_results\models\
    VGG16_best.pth
    ResNet50_best.pth
    MobileNetV3_best.pth
    ... (12 files total)
```

You can change this folder path in the app's sidebar at runtime.

---

## Setup (one time)

### Step 1 — Install PyTorch with CUDA 12.8 (for your RTX 5060)

Open PowerShell or CMD and run:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> If you already installed PyTorch for the training notebook, skip this step —
> it's the same installation.

### Step 2 — Install the remaining dependencies

From inside the `webapp/` folder:

```
pip install -r requirements.txt
```

---

## Running the app

From inside the `webapp/` folder:

```
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`. If it doesn't,
open that address manually.

On first launch the app loads all 12 models into memory (a few seconds on GPU,
up to ~30 seconds on CPU). This happens only once per session.

---

## Using the app

1. **Check the sidebar.** It shows how many models loaded successfully. If it
   says `0 / 12`, fix the "Trained models folder" path to point at your `.pth`
   files.
2. **Single Image tab** — drag in a histopathology image. You'll see:
   - The consensus prediction (majority vote of 12 models)
   - A table of each model's individual prediction and confidence
   - A probability matrix (12 models × 5 classes)
   - A Grad-CAM heatmap (pick a model from the dropdown)
3. **Batch Analysis tab** — drag in multiple images, click "Run batch analysis",
   then download the CSV of results.
4. **History tab** — review and export everything classified this session.

---

*For research and educational use only — not a medical diagnostic device.*
