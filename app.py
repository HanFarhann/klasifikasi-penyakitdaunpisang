import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io

# ====== KONFIGURASI ======
MODEL_PATH = "best_model.h5"

CLASS_NAMES = [
    'bukan_daun_pisang',
    'cordana',
    'healthy',
    'pestalotiopsis',
    'sigatoka'
]

DISEASE_INFO = {
    'cordana': {
        'description': "Penyakit Cordana disebabkan oleh jamur Cordana musae. Gejalanya berupa bercak kuning pucat atau coklat berbentuk oval seperti mata pada daun.",
        'prevention': "Jaga kebersihan kebun, lakukan pemangkasan daun yang terinfeksi secara teratur, dan pastikan jarak tanam tidak terlalu rapat untuk sirkulasi udara yang baik. Penggunaan fungisida berbahan aktif mankozeb atau propikonazol bisa menjadi pilihan."
    },
    'pestalotiopsis': {
        'description': "Penyakit ini disebabkan oleh jamur Pestalotiopsis. Gejalanya adalah bercak kecil berwarna coklat kehitaman pada tepi daun yang kemudian menyebar ke tengah.",
        'prevention': "Hindari luka mekanis pada tanaman. Lakukan sanitasi kebun dengan membuang daun-daun kering dan terinfeksi. Perbaiki drainase tanah dan gunakan fungisida yang mengandung tembaga oksiklorida atau klorotalonil."
    },
    'sigatoka': {
        'description': "Sigatoka adalah salah satu penyakit paling merusak pada pisang, disebabkan oleh jamur Mycosphaerella. Gejalanya berupa garis-garis kecil kuning yang berkembang menjadi bercak coklat dengan tepi kuning.",
        'prevention': "Gunakan varietas pisang yang tahan. Atur drainase dan irigasi dengan baik. Lakukan pemupukan berimbang, terutama Kalium. Semprotkan fungisida sistemik seperti propikonazol atau tebukonazol secara berkala."
    },
    'healthy': {
        'description': "Daun pisang dalam kondisi sehat, tidak menunjukkan gejala penyakit.",
        'prevention': "Pertahankan praktik agronomi yang baik: pemupukan seimbang, irigasi cukup, dan sanitasi kebun untuk menjaga tanaman tetap sehat dan produktif."
    },
    'bukan_daun_pisang': {
        'description': "Gambar yang diunggah tidak terdeteksi sebagai daun pisang atau kualitas gambar tidak cukup baik untuk dianalisis.",
        'prevention': "Silakan coba lagi dengan gambar daun pisang yang lebih jelas dan fokus. Pastikan objek utama dalam gambar adalah daun pisang."
    }
}

# ====== FUNGSI ======
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

def preprocess_image(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

# ====== UI STREAMLIT ======
st.set_page_config(page_title="🍃 Klasifikasi Penyakit Daun Pisang", layout="wide")

st.markdown("<h1 style='text-align:center;'>🍃 Klasifikasi Penyakit Daun Pisang</h1>", unsafe_allow_html=True)
st.write("Unggah gambar daun pisang untuk memprediksi jenis penyakitnya.")

model = load_model()

uploaded_file = st.file_uploader("📤 Pilih gambar daun pisang", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model:
    image_bytes = uploaded_file.read()

    # Proses prediksi
    with st.spinner("🔍 Sedang memproses..."):
        processed_image = preprocess_image(image_bytes)
        prediction = model.predict(processed_image)
        confidence = np.max(prediction[0])
        predicted_class_index = np.argmax(prediction[0])
        class_name = CLASS_NAMES[predicted_class_index]

    info = DISEASE_INFO.get(class_name, {
        'description': 'Informasi tidak ditemukan.',
        'prevention': 'Tidak ada saran penanganan.'
    })

    # Layout 2 kolom: Gambar di kiri, hasil di kanan
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image_bytes, caption="Gambar yang diunggah", use_container_width=True)

    with col2:
        if class_name == 'bukan_daun_pisang':
            st.error("### ❌ Hasil Prediksi: Bukan Daun Pisang.")
        else:
            st.success(f"### ✅ Hasil Prediksi: **{class_name.replace('_', ' ').title()}**")
        st.write(f"📊 **Tingkat Kepercayaan (Confidence):** {confidence*100:.2f}%")
        st.markdown(f"### 📋 Deskripsi")
        st.write(info['description'])
        st.markdown(f"### 🛡 Pencegahan")
        st.write(info['prevention'])