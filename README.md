# Leakage-Resistant Benchmarking of Twelve Deep Learning Architectures on LC25000

Code, recovered source-group assignments and results for the manuscript
*Leakage-Resistant Benchmarking of Twelve Deep Learning Architectures for Lung and
Colon Cancer Histopathology Classification, with Parameter-Efficient Hybrid Adapters.*

---

## Summary

LC25000's 25,000 images derive from only **1,250 original biopsies** through rotation
and flip augmentation, and the public release does not encode source identity.
Conventional image-level partitioning therefore places augmented derivatives of the
same biopsy in both the training and test sets.

We recover the latent source structure, quantify the resulting leakage, and re-evaluate
twelve architectures under a group-aware partition.

| Finding | Evidence |
|---|---|
| Near-duplicate test contamination | **48.76% → 0.00%** at cosine similarity > 0.98 |
| Source groups recovered | **1,221 / 1,250** (97.7%) |
| Ceiling effect | **9/12 → 0/12** models above 99.8% accuracy |
| Statistical separability | All **15/15** pairwise 95% CIs overlap across 5 repeated partitions |
| Clean accuracy vs robustness | Spearman **ρ = 0.10, p = 0.75** between rankings |

**The recovered group assignments are released so that leakage-resistant evaluation on
LC25000 can be adopted directly** — see `results/partitioning/lc25000_final_split.csv`.

---

## Repository structure

```
notebooks/
  LC25000_Complete_Pipeline.ipynb   Full reproducible pipeline (26 sections)
webapp/
  app.py, models.py, gradcam.py     Interactive Streamlit classifier
scripts/
  collect_results.py                Copies result files into results/
  build_supplementary.py            Generates Supplementary Tables S1–S20
  build_supplementary_figures.py    Generates Supplementary Figures S1–S12
results/
  partitioning/                     Group assignments, threshold sweep, leakage
  leakage_resistant/                Protocol B metrics, CIs, McNemar, per-class
  multiseed/                        Five repeated group-aware partitions
  ablations/                        12 component-ablation variants
  regimes/                          Matched frozen / partial / full fine-tuning
  jitter_sweep/                     Perturbation-intensity calibration
  cost/                             FLOPs, memory, throughput, storage
  gating/                           DPCT-Net feature-fusion weights
  gradcam/                          Inter-model attention agreement
  lunghist700/                      External-dataset transfer and fine-tuning
figures/                            All manuscript and supplementary figures
docs/                               Mathematical formulation of both architectures
```

---

## Using the leakage-resistant partition

To reproduce our partition without rerunning the clustering:

```python
import pandas as pd
split = pd.read_csv("results/partitioning/lc25000_final_split.csv")

train = split[split.fold == "train"]   # 15,992 images
val   = split[split.fold == "val"]     #  4,002 images
test  = split[split.fold == "test"]    #  5,006 images
```

Columns: `path`, `class`, `group_id` (initial cluster), `group_final` (after merging),
`fold`. Every image sharing a `group_final` derives from the same source biopsy and is
confined to one fold.

---

## Reproducing the analysis

**Requirements** — Python 3.12, CUDA-capable GPU (developed on an RTX 5060, CUDA 12.8).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install timm scikit-learn pandas matplotlib seaborn pillow tqdm scipy jupyter
```

Open `notebooks/LC25000_Complete_Pipeline.ipynb`, edit the paths in Section 0, and run
from the top. The notebook is checkpoint-resume aware: completed models are loaded
rather than retrained.

> **Windows note.** `NUM_WORKERS` must remain 0; spawn-based multiprocessing crashes
> PyTorch DataLoader workers on Windows.

### Pipeline sections

| Sections | Content |
|---|---|
| 0–2 | Configuration, environment, dataset verification |
| **3** | **Leakage analysis, group recovery, threshold sweep, partitioning** |
| 4–6 | Transforms, twelve architectures, training utilities |
| 7–9 | Training under both protocols, dual-protocol comparison |
| 10–16 | Metrics, TP/FP/TN/FN, robustness, efficiency, LungHist700, statistics |
| **17–22** | **Ablations, repeated partitions, matched regimes, fusion weights, Grad-CAM** |
| **23–26** | **FLOPs and memory, intensity calibration, proposed-model regimes** |

---

## Architectures

Twelve models across four families, all trained under a unified evaluation protocol.
Training regime and trainable fraction are reported per model, since these differ by
design and are a stronger determinant of accuracy than architecture (see manuscript
Section 5.2).

| Family | Models |
|---|---|
| Classical CNN | VGG-16, ResNet-50, DenseNet-121 |
| Efficient CNN | MobileNetV3-Small, EfficientNet-B0, EfficientNet-B3 |
| Modern / Transformer | ConvNeXt-Tiny, ViT-Base/16, Swin-Tiny |
| Hybrid adapters | FSPAN-Y, **DPCT-Net**, **MSCA-Net** |

**DPCT-Net** — frozen ConvNeXt-Tiny and Swin-Tiny streams fused by cross-attention and
per-sample gating. 56.13 M total, 0.791 M trainable (1.41%).

**MSCA-Net** — frozen ConvNeXt-Tiny with multi-scale top-down cascade attention.
28.41 M total, 0.596 M trainable (2.10%).

Architecture diagrams: `figures/DPCT_Net_architecture.png`, `figures/MSCA_Net_architecture.png`.
Formal definitions with all equations: `docs/LC25000_Mathematical_Formulation.docx`.

> Trainable-parameter count measures **adaptation** cost, not inference cost. DPCT-Net
> trains fewer parameters than MobileNetV3 but requires 161× the FLOPs — see
> `results/cost/computational_cost.csv`.

---

## Interactive web application

```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py
```

Loads trained checkpoints, classifies uploaded images with all twelve models, and
produces Grad-CAM overlays. See `webapp/README.md`.

---

## Model checkpoints

Trained weights (~1.5 GB) are not stored in this repository. They are available at
**[DOI / release link — to be added]**.

---

## Datasets

- **LC25000** — Borkowski et al. (2019). Publicly available, fully de-identified.
- **LungHist700** — 691 images, 45 patients. Publicly available, fully de-identified.

Neither dataset is redistributed here.

---

## Citation

```bibtex
@article{asaduzzaman2026leakage,
  title   = {Leakage-Resistant Benchmarking of Twelve Deep Learning Architectures
             for Lung and Colon Cancer Histopathology Classification,
             with Parameter-Efficient Hybrid Adapters},
  author  = {Asaduzzaman and Hien, Nguyen Dinh and Das, Utpol Kanti and
             Chen, Chang-I and Chang, Cheng-Jen and Yang, Tzu-Sen},
  journal = {Pattern Analysis and Applications},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Code released under the MIT License (see `LICENSE`). Dataset terms are those of the
original providers.

## Contact

Tzu-Sen Yang — tsyang@tmu.edu.tw
Graduate Institute of Biomedical Materials and Tissue Engineering, Taipei Medical University
