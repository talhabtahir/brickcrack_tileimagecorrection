import io
import math
import os
import zipfile
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from keras.models import Model

# ============================================================
# CONFIG
# ============================================================

TILE_SIZE = 224
ANNOTATION_DIR = "tile_annotations"

Path(ANNOTATION_DIR).mkdir(exist_ok=True)
Path(f"{ANNOTATION_DIR}/tiles").mkdir(exist_ok=True)

CLASS_LABELS = {
    0: "Normal",
    1: "Cracked",
    2: "Not a Wall"
}

CLASS_COLORS_BGR = {
    0: (0, 200, 0),
    1: (0, 0, 255),
    2: (0, 165, 255),
}

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Tile Annotation Correction Tool",
    layout="wide"
)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    import h5py
    from tensorflow.keras.models import model_from_json

    # Try normal load first
    try:
        return tf.keras.models.load_model(
            "170kmodelv10_version_cam_1.keras",
            compile=False
        )
    except:
        pass

    # Fallback: manually reconstruct model
    with h5py.File("170kmodelv10_version_cam_1.keras", "r") as f:
        model_json = f.attrs["model_config"].decode("utf-8")
        model = model_from_json(model_json)
        model.load_weights(f["model_weights"])

    model.trainable = False
    return model



@st.cache_resource
def build_custom_model(sensitivity: int):
    model = load_model()

    return Model(
        inputs=model.inputs,
        outputs=(
            model.layers[sensitivity].output,
            model.layers[-1].output,
        ),
    )


# ============================================================
# TILE PREDICTION
# ============================================================

def predict_tiles_batch(tiles_np, sensitivity=9):

    batch = np.stack(tiles_np, axis=0).astype(np.float32) / 255.0

    custom_model = build_custom_model(sensitivity)

    conv_outputs, pred_vecs = custom_model.predict(
        batch,
        verbose=0,
    )

    pred_indices = [
        int(np.argmax(pv))
        for pv in pred_vecs
    ]

    return pred_indices, pred_vecs, conv_outputs


# ============================================================
# TILE ANALYSIS
# ============================================================

def tiled_crack_detection(
    image_bytes: bytes,
    sensitivity: int = 9,
    confidence_threshold: float = 50.0,
):

    image_hash_short = hashlib.md5(image_bytes).hexdigest()[:12]

    image_data = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    original_img = np.array(image_data)

    orig_h, orig_w, _ = original_img.shape

    pad_h = (TILE_SIZE - orig_h % TILE_SIZE) % TILE_SIZE
    pad_w = (TILE_SIZE - orig_w % TILE_SIZE) % TILE_SIZE

    padded_img = cv2.copyMakeBorder(
        original_img,
        0,
        pad_h,
        0,
        pad_w,
        cv2.BORDER_REFLECT,
    )

    pad_h_total, pad_w_total = padded_img.shape[:2]

    n_rows = pad_h_total // TILE_SIZE
    n_cols = pad_w_total // TILE_SIZE

    numbered_canvas = padded_img.copy()
    tile_grid_overlay = padded_img.copy().astype(np.float32)

    tile_coords = []
    tiles_np = []

    for r in range(n_rows):
        for c in range(n_cols):

            y0 = r * TILE_SIZE
            y1 = (r + 1) * TILE_SIZE

            x0 = c * TILE_SIZE
            x1 = (c + 1) * TILE_SIZE

            tile_coords.append((r, c, y0, y1, x0, x1))

            tile = padded_img[y0:y1, x0:x1]

            tiles_np.append(tile)

    pred_indices, pred_vecs, conv_outputs = predict_tiles_batch(
        tiles_np,
        sensitivity,
    )

    tile_results = []

    font = cv2.FONT_HERSHEY_SIMPLEX

    for tile_idx, (r, c, y0, y1, x0, x1) in enumerate(tile_coords):

        pred_index = pred_indices[tile_idx]
        pred_vec = pred_vecs[tile_idx]

        confidence = float(pred_vec[pred_index]) * 100

        if pred_index == 1 and confidence < confidence_threshold:
            pred_index = 0

        label_name = CLASS_LABELS[pred_index]

        tile_filename = f"{image_hash_short}_r{r}_c{c}.png"

        tile_img = padded_img[y0:y1, x0:x1]

        Image.fromarray(tile_img).save(
            f"{ANNOTATION_DIR}/tiles/{tile_filename}"
        )

        tile_results.append({
            "tile_id": tile_idx,
            "filename": tile_filename,
            "row": r,
            "col": c,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "predicted_label": label_name,
            "confidence": round(confidence, 2),
        })

        color_bgr = CLASS_COLORS_BGR[pred_index]
        color_rgb = color_bgr[::-1]

        alpha = 0.35

        tile_grid_overlay[y0:y1, x0:x1] = (
            (1 - alpha)
            * tile_grid_overlay[y0:y1, x0:x1]
            + alpha * np.array(color_rgb)
        )

        cv2.rectangle(
            numbered_canvas,
            (x0, y0),
            (x1 - 1, y1 - 1),
            color_rgb,
            2,
        )

        tile_number = str(tile_idx)

        font_scale = 0.7
        font_thickness = 2

        (tw, th), _ = cv2.getTextSize(
            tile_number,
            font,
            font_scale,
            font_thickness,
        )

        tx = x0 + 8
        ty = y0 + th + 8

        cv2.putText(
            numbered_canvas,
            tile_number,
            (tx + 1, ty + 1),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness + 1,
            cv2.LINE_AA,
        )

        cv2.putText(
            numbered_canvas,
            tile_number,
            (tx, ty),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    numbered_image = Image.fromarray(
        numbered_canvas[:orig_h, :orig_w]
    )

    tile_grid_image = Image.fromarray(
        tile_grid_overlay.astype(np.uint8)[:orig_h, :orig_w]
    )

    summary = {
        "tiles": tile_results,
        "total": len(tile_results),
        "cracked": sum(
            1 for t in tile_results
            if t["predicted_label"] == "Cracked"
        ),
        "normal": sum(
            1 for t in tile_results
            if t["predicted_label"] == "Normal"
        ),
        "not_wall": sum(
            1 for t in tile_results
            if t["predicted_label"] == "Not a Wall"
        ),
    }

    return tile_grid_image, numbered_image, summary


# ============================================================
# UI
# ============================================================

st.title("Semi-Automatic Tile Annotation Correction Tool")

st.markdown(
    """
This tool:
- predicts tile classes automatically
- shows tile grid localization
- allows correction of labels
- exports CSV + tile dataset
"""
)

file = st.file_uploader(
    "Upload wall image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if file is not None:

    image_bytes = file.getvalue()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    st.subheader("Original Image")

    st.image(image, use_container_width=True)

    st.divider()

    sensitivity = st.slider(
        "Sensitivity",
        0,
        12,
        9,
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        10.0,
        99.0,
        50.0,
    )

    if st.button("Run Tile Analysis"):

        with st.spinner("Running tile prediction..."):

            tile_grid_img, numbered_img, summary = tiled_crack_detection(
                image_bytes,
                sensitivity=sensitivity,
                confidence_threshold=confidence_threshold,
            )

        st.session_state["summary"] = summary
        st.session_state["tile_grid_img"] = tile_grid_img
        st.session_state["numbered_img"] = numbered_img

if "summary" in st.session_state:

    summary = st.session_state["summary"]

    st.divider()

    st.subheader("Tile Analysis Results")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Total Tiles", summary["total"])
    m2.metric("Cracked", summary["cracked"])
    m3.metric("Normal", summary["normal"])
    m4.metric("Not Wall", summary["not_wall"])

    c1, c2 = st.columns(2)

    with c1:
        st.image(
            st.session_state["tile_grid_img"],
            caption="Color-Coded Tile Grid",
            use_container_width=True,
        )

    with c2:
        st.image(
            st.session_state["numbered_img"],
            caption="Numbered Tile Grid",
            use_container_width=True,
        )

    st.divider()

    st.subheader("Tile Annotation Correction")

    st.markdown(
        "Review predictions and correct wrong labels"
    )

    corrected_results = []

    tile_cols = st.columns(4)

    for idx, tile_info in enumerate(summary["tiles"]):

        col = tile_cols[idx % 4]

        tile_path = (
            f"{ANNOTATION_DIR}/tiles/{tile_info['filename']}"
        )

        with col:

            st.image(
                tile_path,
                caption=f"Tile {tile_info['tile_id']}",
                use_container_width=True,
            )

            st.write(
                f"Prediction: {tile_info['predicted_label']}"
            )

            st.write(
                f"Confidence: {tile_info['confidence']}%"
            )

            labels = [
                "Normal",
                "Cracked",
                "Not a Wall",
            ]

            corrected_label = st.radio(
                f"Correct Label #{idx}",
                labels,
                index=labels.index(
                    tile_info["predicted_label"]
                ),
                key=f"tile_{idx}",
            )

            corrected_results.append({
                "tile_id": tile_info["tile_id"],
                "filename": tile_info["filename"],
                "row": tile_info["row"],
                "col": tile_info["col"],
                "x0": tile_info["x0"],
                "y0": tile_info["y0"],
                "x1": tile_info["x1"],
                "y1": tile_info["y1"],
                "predicted_label": tile_info[
                    "predicted_label"
                ],
                "corrected_label": corrected_label,
                "confidence": tile_info["confidence"],
            })

    st.divider()

    st.subheader("Export Dataset")

    if st.button("Export Corrected CSV"):

        corrected_df = pd.DataFrame(corrected_results)

        csv_path = (
            f"{ANNOTATION_DIR}/corrected_tile_annotations.csv"
        )

        corrected_df.to_csv(csv_path, index=False)

        st.success("CSV exported successfully")

        st.dataframe(corrected_df.head())

        with open(csv_path, "rb") as f:
            st.download_button(
                label="Download CSV",
                data=f,
                file_name="corrected_tile_annotations.csv",
                mime="text/csv",
            )

    if st.button("Export ZIP Dataset"):

        corrected_df = pd.DataFrame(corrected_results)

        csv_path = (
            f"{ANNOTATION_DIR}/corrected_tile_annotations.csv"
        )

        corrected_df.to_csv(csv_path, index=False)

        zip_path = f"{ANNOTATION_DIR}/tile_dataset.zip"

        with zipfile.ZipFile(zip_path, "w") as zipf:

            zipf.write(
                csv_path,
                arcname="corrected_tile_annotations.csv",
            )

            for tile_info in corrected_results:

                tile_path = (
                    f"{ANNOTATION_DIR}/tiles/"
                    f"{tile_info['filename']}"
                )

                if os.path.exists(tile_path):

                    zipf.write(
                        tile_path,
                        arcname=f"tiles/{tile_info['filename']}"
                    )

        st.success("ZIP dataset created successfully")

        with open(zip_path, "rb") as f:
            st.download_button(
                label="Download ZIP Dataset",
                data=f,
                file_name="tile_dataset.zip",
                mime="application/zip",
            )


    # ============================================================
    # SHOW FINAL DATAFRAME
    # ============================================================

    st.divider()

    st.subheader("Current Annotation Table")

    current_df = pd.DataFrame(corrected_results)

    st.dataframe(
        current_df,
        use_container_width=True,
    )

    # ============================================================
    # OPTIONAL FILTERS
    # ============================================================

    st.divider()

    st.subheader("Annotation Statistics")

    total_tiles = len(current_df)

    cracked_tiles = len(
        current_df[
            current_df["corrected_label"] == "Cracked"
        ]
    )

    normal_tiles = len(
        current_df[
            current_df["corrected_label"] == "Normal"
        ]
    )

    notwall_tiles = len(
        current_df[
            current_df["corrected_label"] == "Not a Wall"
        ]
    )

    s1, s2, s3, s4 = st.columns(4)

    s1.metric("Total Tiles", total_tiles)
    s2.metric("Cracked", cracked_tiles)
    s3.metric("Normal", normal_tiles)
    s4.metric("Not Wall", notwall_tiles)

    # ============================================================
    # AGREEMENT ANALYSIS
    # ============================================================

    st.divider()

    st.subheader("Model vs Human Agreement")

    agreement_count = len(
        current_df[
            current_df["predicted_label"]
            ==
            current_df["corrected_label"]
        ]
    )

    disagreement_count = total_tiles - agreement_count

    agreement_pct = (
        agreement_count / total_tiles * 100
        if total_tiles > 0 else 0
    )

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "Agreement",
        agreement_count
    )

    a2.metric(
        "Corrections Needed",
        disagreement_count
    )

    a3.metric(
        "Agreement %",
        f"{agreement_pct:.2f}%"
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Semi-Automatic Tile Annotation Tool "
    "for Crack Localization Research"
)
