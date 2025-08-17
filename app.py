import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# 1. Bangun ulang arsitektur sesuai training
# =========================
IMG_SIZE = (224, 224)  # ganti kalau kamu pakai ukuran lain
NUM_CLASSES = 5        # ganti sesuai jumlah kelasmu

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights=None
)

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# =========================
# 2. Load weights hasil training
# =========================
model.load_weights("best_model.h5")

# =========================
# 3. Streamlit UI
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
