import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# =========================
# 1. Load model langsung
# =========================
MODEL_PATH = "best_model.h5"
model = load_model(MODEL_PATH)

# Kalau kamu perlu tahu jumlah kelas:
NUM_CLASSES = model.output_shape[-1]
IMG_SIZE = model.input_shape[1:3]

# =========================
# 2. Streamlit UI
# =========================
st.title("🍌 Klasifikasi Penyakit Daun Pisang")

uploaded_file = st.file_uploader("Unggah gambar daun pisang", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Buka gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar diunggah", use_column_width=True)

    # Preprocessing
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)

    # Prediksi
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])

    st.write(f"### ✅ Prediksi: Kelas {class_idx} (confidence: {confidence:.2f})")
