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
    model = tf.keras.models.load_model(
        "170kmodelv10_version_cam_1.keras"
    )
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
st.dataframe(corrected_df.head())
with open(csv_path, "rb") as f:
		st.download_button(
			label="Download CSV",
			data=f,
			file_name="corrected_tile_annotations.csv",
			mime="text/csv",
		)

# ============================================================
# EXPORT ZIP DATASET
# ============================================================

if st.button("Export ZIP Dataset"):

	corrected_df = pd.DataFrame(corrected_results)

	csv_path = (
		f"{ANNOTATION_DIR}/corrected_tile_annotations.csv"
	)

	corrected_df.to_csv(csv_path, index=False)

	zip_path = f"{ANNOTATION_DIR}/tile_dataset.zip"

	with zipfile.ZipFile(zip_path, "w") as zipf:

		# Add CSV
		zipf.write(
			csv_path,
			arcname="corrected_tile_annotations.csv",
		)

		# Add all tile images
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
