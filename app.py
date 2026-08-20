"""
app.py
======
LC25000 Histopathology Classifier — Streamlit web application.

Run locally with:
    streamlit run app.py

Features:
  - Tab 1 "Single Image": upload one image, see predictions from ALL 12 models
    side-by-side, a consensus vote, a probability matrix, and Grad-CAM heatmaps.
  - Tab 2 "Batch Analysis": upload many images, get a results table + CSV download.
  - Tab 3 "History": every prediction made this session, with a clear button.

This file contains the Streamlit UI. The model architectures live in models.py
and the Grad-CAM logic lives in gradcam.py.
"""

import io
import time
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import streamlit as st
from PIL import Image

import models as M
import gradcam as G


# ======================================================================
# PAGE CONFIG — must be the first Streamlit call
# ======================================================================
st.set_page_config(
    page_title="LC25000 Histopathology Classifier",
    page_icon="🔬",
    layout="wide",
)


# ======================================================================
# CONFIGURATION
# ======================================================================
# Default folder containing your trained checkpoints ({ModelName}_best.pth).
# You can also change this live in the sidebar.
DEFAULT_WEIGHTS_DIR = r"E:\LC25000_results\models"


# ======================================================================
# DEVICE — use GPU if available, otherwise CPU (so it works after deployment)
# ======================================================================
@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# MODEL LOADING — cached so models load only ONCE per session, not on
# every interaction. st.cache_resource is the correct cache for objects
# like ML models (as opposed to st.cache_data for serializable data).
# ======================================================================
@st.cache_resource(show_spinner=False)
def load_all_models(weights_dir):
    """
    Try to load every model in the registry from `weights_dir`.
    Returns: (loaded_models dict, status list of (name, ok, message)).
    """
    device = get_device()
    weights_dir = Path(weights_dir)
    loaded = {}
    status = []
    for name in M.MODEL_REGISTRY:
        ckpt = weights_dir / f"{name}_best.pth"
        if not ckpt.exists():
            status.append((name, False, f"Checkpoint not found: {ckpt.name}"))
            continue
        try:
            model = M.load_model(name, str(ckpt), device)
            loaded[name] = model
            status.append((name, True, "Loaded"))
        except Exception as e:
            status.append((name, False, f"{type(e).__name__}: {e}"))
    return loaded, status


# ======================================================================
# PREDICTION HELPERS
# ======================================================================
def predict_all_models(models_dict, image_tensor, device):
    """
    Run every loaded model on one image.
    Returns a dict: model_name -> probability array of length 5.
    """
    results = {}
    for name, model in models_dict.items():
        probs = M.predict(model, image_tensor, device)
        results[name] = probs
    return results


def consensus_vote(results):
    """
    Given {model: probs}, return (majority_class, agreement_count, total).
    """
    votes = [M.CLASS_NAMES[int(np.argmax(p))] for p in results.values()]
    if not votes:
        return None, 0, 0
    # most common vote
    unique, counts = np.unique(votes, return_counts=True)
    winner_idx = int(np.argmax(counts))
    return unique[winner_idx], int(counts[winner_idx]), len(votes)


# ======================================================================
# SIDEBAR — configuration and model status
# ======================================================================
st.sidebar.title("🔬 LC25000 Classifier")
st.sidebar.markdown("Lung & colon cancer histopathology — 12-model comparison - Developed by Zaman.")

weights_dir = st.sidebar.text_input(
    "Trained models folder",
    value=DEFAULT_WEIGHTS_DIR,
    help="Folder containing your {ModelName}_best.pth files.",
)

device = get_device()
st.sidebar.markdown(f"**Compute device:** `{device.type.upper()}`")

# Load models (cached). The spinner only shows on the first run.
with st.spinner("Loading trained models — this happens once..."):
    MODELS, STATUS = load_all_models(weights_dir)

# Show load status in the sidebar
n_ok = sum(1 for _, ok, _ in STATUS if ok)
st.sidebar.markdown(f"**Models loaded:** {n_ok} / {len(M.MODEL_REGISTRY)}")
with st.sidebar.expander("Model load details", expanded=(n_ok == 0)):
    for name, ok, msg in STATUS:
        icon = "✅" if ok else "❌"
        st.write(f"{icon} **{name}** — {msg}")

if n_ok == 0:
    st.error(
        "No models could be loaded. Check that the 'Trained models folder' path "
        "in the sidebar is correct and contains your `{ModelName}_best.pth` files."
    )
    st.stop()

# Class reference in the sidebar
with st.sidebar.expander("The 5 diagnostic classes"):
    for key, label in M.CLASS_LABELS.items():
        st.write(f"**{key}** — {label}")

# Session-state history store (persists across reruns within a session)
if "history" not in st.session_state:
    st.session_state["history"] = []


# ======================================================================
# MAIN AREA — three tabs
# ======================================================================
st.title("TSY LAB Histopathology Image Classifier")
st.caption(
    "Upload a lung or colon histopathology image and compare predictions "
    "from twelve deep learning models trained on the LC25000 dataset."
)

tab_single, tab_batch, tab_history = st.tabs(
    ["🖼️ Single Image", "📚 Batch Analysis", "🕘 History"]
)


# ----------------------------------------------------------------------
# TAB 1: SINGLE IMAGE
# ----------------------------------------------------------------------
with tab_single:
    st.subheader("Single Image Analysis")
    uploaded = st.file_uploader(
        "Upload a histopathology image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="single_uploader",
    )

    if uploaded is not None:
        # ---- Load & preprocess the image ----
        image = Image.open(uploaded).convert("RGB")
        transform = M.get_transform()
        image_tensor = transform(image)   # (3, 224, 224)

        col_img, col_pred = st.columns([1, 2])

        with col_img:
            st.image(image, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

        # ---- Run all 12 models ----
        with st.spinner("Running all 12 models..."):
            t0 = time.time()
            results = predict_all_models(MODELS, image_tensor, device)
            elapsed = time.time() - t0

        with col_pred:
            # ---- Consensus box ----
            winner, agree, total = consensus_vote(results)
            st.markdown(f"### 🏆 Consensus: `{winner}`")
            st.markdown(f"**{M.CLASS_LABELS.get(winner, winner)}**")
            st.progress(agree / total)
            st.caption(
                f"{agree} of {total} models agree · "
                f"all models ran in {elapsed:.2f}s on {device.type.upper()}"
            )

        # ---- Per-model results table ----
        st.markdown("#### Predictions from each model")
        rows = []
        for name in MODELS:
            probs = results[name]
            pred_idx = int(np.argmax(probs))
            rows.append({
                "Model": name,
                "Family": M.MODEL_FAMILY.get(name, ""),
                "Prediction": M.CLASS_NAMES[pred_idx],
                "Confidence": f"{probs[pred_idx] * 100:.2f}%",
                "Agrees with consensus": "✅" if M.CLASS_NAMES[pred_idx] == winner else "—",
            })
        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # ---- Probability matrix (12 models x 5 classes) ----
        st.markdown("#### Probability matrix")
        st.caption("Each cell is the probability that a model assigns to a class. Darker = higher.")
        prob_matrix = pd.DataFrame(
            {name: results[name] for name in MODELS},
            index=M.CLASS_NAMES,
        ).T  # transpose so rows = models, columns = classes
        st.dataframe(
            prob_matrix.style
                .format("{:.3f}")
                .background_gradient(cmap="Blues", axis=None),
            use_container_width=True,
        )

        # ---- Grad-CAM section ----
        st.markdown("#### 🔥 Grad-CAM — where the model is looking")
        st.caption(
            "Grad-CAM highlights the image regions that most influenced the "
            "prediction. Available for convolutional models; pure transformers "
            "(ViT-Base, Swin-Tiny) and DPCT-Net are not supported."
        )
        gradcam_model = st.selectbox(
            "Choose a model for Grad-CAM",
            options=list(MODELS.keys()),
            key="gradcam_select",
        )
        if gradcam_model in G.GRADCAM_SUPPORTED:
            with st.spinner(f"Computing Grad-CAM for {gradcam_model}..."):
                pred_idx = int(np.argmax(results[gradcam_model]))
                overlay = G.compute_gradcam(
                    MODELS[gradcam_model], gradcam_model,
                    image_tensor, pred_idx, device,
                )
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(image, caption="Original", use_container_width=True)
            with gc2:
                st.image(
                    overlay,
                    caption=f"Grad-CAM: {gradcam_model} → {M.CLASS_NAMES[pred_idx]}",
                    use_container_width=True,
                )
        else:
            reason = G.GRADCAM_UNSUPPORTED_REASON.get(gradcam_model, "Not supported.")
            st.info(f"Grad-CAM not available for **{gradcam_model}**. {reason}")

        # ---- Save to history ----
        st.session_state["history"].append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "filename": uploaded.name,
            "consensus": winner,
            "agreement": f"{agree}/{total}",
            **{name: M.CLASS_NAMES[int(np.argmax(results[name]))] for name in MODELS},
        })


# ----------------------------------------------------------------------
# TAB 2: BATCH ANALYSIS
# ----------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch Analysis")
    st.caption(
        "Upload several images at once. Each image is classified by all 12 "
        "models and the consensus is reported. Results can be downloaded as CSV."
    )
    batch_files = st.file_uploader(
        "Upload multiple histopathology images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        st.write(f"**{len(batch_files)} images uploaded.**")
        run_batch = st.button("▶️ Run batch analysis", type="primary")

        if run_batch:
            transform = M.get_transform()
            batch_rows = []
            progress = st.progress(0.0, text="Starting...")

            for i, f in enumerate(batch_files):
                image = Image.open(f).convert("RGB")
                image_tensor = transform(image)
                results = predict_all_models(MODELS, image_tensor, device)
                winner, agree, total = consensus_vote(results)

                row = {
                    "Filename": f.name,
                    "Consensus": winner,
                    "Agreement": f"{agree}/{total}",
                }
                # add each model's prediction + confidence
                for name in MODELS:
                    probs = results[name]
                    idx = int(np.argmax(probs))
                    row[f"{name}"] = M.CLASS_NAMES[idx]
                    row[f"{name}_conf"] = round(float(probs[idx]), 4)
                batch_rows.append(row)

                # also log to history
                st.session_state["history"].append({
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "filename": f.name,
                    "consensus": winner,
                    "agreement": f"{agree}/{total}",
                    **{name: M.CLASS_NAMES[int(np.argmax(results[name]))] for name in MODELS},
                })

                progress.progress((i + 1) / len(batch_files),
                                  text=f"Processed {i + 1}/{len(batch_files)}")

            progress.empty()
            batch_df = pd.DataFrame(batch_rows)
            st.success(f"Done — {len(batch_files)} images classified.")

            # Show a compact view (filename, consensus, agreement) + full table in expander
            st.markdown("#### Summary")
            st.dataframe(
                batch_df[["Filename", "Consensus", "Agreement"]],
                use_container_width=True, hide_index=True,
            )
            with st.expander("Full results (every model, every image)"):
                st.dataframe(batch_df, use_container_width=True, hide_index=True)

            # CSV download
            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results as CSV",
                data=csv_bytes,
                file_name="lc25000_batch_results.csv",
                mime="text/csv",
            )

            # Quick class-distribution chart of the consensus predictions
            st.markdown("#### Consensus class distribution")
            dist = batch_df["Consensus"].value_counts()
            st.bar_chart(dist)


# ----------------------------------------------------------------------
# TAB 3: HISTORY
# ----------------------------------------------------------------------
with tab_history:
    st.subheader("Prediction History (this session)")
    history = st.session_state["history"]

    if not history:
        st.info("No predictions yet. Use the Single Image or Batch Analysis tabs.")
    else:
        st.write(f"**{len(history)} predictions made this session.**")
        hist_df = pd.DataFrame(history)
        # Put the summary columns first
        front = ["time", "filename", "consensus", "agreement"]
        cols = front + [c for c in hist_df.columns if c not in front]
        hist_df = hist_df[cols]
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download history as CSV",
                data=csv_bytes,
                file_name="lc25000_prediction_history.csv",
                mime="text/csv",
            )
        with c2:
            if st.button("🗑️ Clear history"):
                st.session_state["history"] = []
                st.rerun()


# ======================================================================
# FOOTER
# ======================================================================
st.markdown("---")
st.caption(
    "LC25000 Histopathology Classifier · Built with Streamlit and PyTorch · "
    "For research and educational use only — not a medical diagnostic device."
)
