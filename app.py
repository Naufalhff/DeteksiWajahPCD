import streamlit as st
import os
import numpy as np
from PIL import Image
import pandas as pd
import cv2
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import torch
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from mtcnn import MTCNN

# Import fungsi dari scripts
from scripts.embed_faces import get_face_embedding
from scripts.face_similarity_faces import get_similarity

# ===== CONFIGURASI =====
MODEL_PATH = 'final_ethnicity_model.h5'  # Model dalam format H5 (TensorFlow)
LABELS = ['Jawa', 'Sunda']  # Sesuaikan dengan label etnis yang kamu pakai
UPLOAD_FOLDER = 'temp_uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===== FUNGSI UTILITAS =====
@st.cache_resource
def load_ethnicity_model():
    """Load model TensorFlow dari file H5"""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_facenet_model():
    """Load model FaceNet untuk ekstraksi fitur wajah"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return model, device

def get_face_embedding(image):
    """Get face embedding from image (PIL Image or file path)"""
    model, device = load_facenet_model()
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    try:
        if isinstance(image, str):
            # If image is a file path
            img = Image.open(image).convert('RGB')
        else:
            # If image is already a PIL Image
            img = image.convert('RGB')
            
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model(img_tensor)
            
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        st.error(f"Error extracting face embedding: {e}")
        return None

def predict_ethnicity(image, model):
    """Predict ethnicity using TensorFlow model"""
    try:
        # Resize gambar ke ukuran yang diharapkan oleh model (160x160)
        img = image.resize((160, 160))
        img_array = np.array(img) / 255.0  # Normalisasi ke [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
        
        # TensorFlow prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        predicted_label = LABELS[predicted_class]
        confidence = prediction[0][predicted_class]
        
        return predicted_label, confidence
    except Exception as e:
        st.error(f"Error predicting ethnicity: {e}")
        return "Wajah tidak terdeteksi", 0

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    if embedding1 is None or embedding2 is None:
        return 0
    
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

def capture_image(key):
    """Capture image from webcam with face detection and ethnicity label"""
    img_file_buffer = st.camera_input("Ambil foto dari kamera", key=key)
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        image_np = np.array(image)  # Konversi ke numpy array
        
        # Inisialisasi MTCNN
        detector = MTCNN()
        
        # Deteksi wajah menggunakan MTCNN
        faces = detector.detect_faces(image_np)
        
        # Gambar kotak dan tambahkan teks untuk setiap wajah yang terdeteksi
        for face in faces:
            x, y, width, height = face['box']
            
            # Gambar kotak biru di sekitar wajah
            cv2.rectangle(image_np, (x, y), (x+width, y+height), (255, 0, 0), 2)
            
            # Prediksi etnis
            face_image = image.crop((x, y, x+width, y+height))
            label, confidence = predict_ethnicity(face_image, load_ethnicity_model())
            
            # Tambahkan teks suku di atas kotak
            cv2.putText(image_np, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Konversi kembali ke PIL Image
        image_with_box = Image.fromarray(image_np)
        return image_with_box
    return None

def save_uploaded_image(uploaded_file):
    """Save uploaded image and return path"""
    if uploaded_file is None:
        return None
    
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# ===== UI FUNCTIONS =====
def ethnicity_detection_ui():
    st.title("👤 Deteksi Etnis Berdasarkan Wajah")
    
    # Tab untuk upload gambar atau gunakan kamera
    tab1, tab2 = st.tabs(["Upload Gambar", "Gunakan Kamera"])
    
    with tab1:
        uploaded_file = st.file_uploader("Unggah gambar wajah", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
            process_ethnicity_detection(image)
    
    with tab2:
        image = capture_image("camera_ethnicity")
        if image:
            process_ethnicity_detection(image)

def face_similarity_ui():
    st.title("🔄 Perbandingan Kemiripan Wajah")
    
    # Tab untuk upload gambar atau gunakan kamera
    tab1, tab2 = st.tabs(["Upload Gambar", "Gunakan Kamera"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Wajah Pertama")
            uploaded_file1 = st.file_uploader("Unggah gambar wajah 1", type=["jpg", "jpeg", "png"], key="face1")
            if uploaded_file1:
                image1 = Image.open(uploaded_file1).convert('RGB')
                st.image(image1, caption='Wajah 1', use_column_width=True)
        
        with col2:
            st.subheader("Wajah Kedua")
            uploaded_file2 = st.file_uploader("Unggah gambar wajah 2", type=["jpg", "jpeg", "png"], key="face2")
            if uploaded_file2:
                image2 = Image.open(uploaded_file2).convert('RGB')
                st.image(image2, caption='Wajah 2', use_column_width=True)
        
        if uploaded_file1 and uploaded_file2:
            process_face_similarity(image1, image2)
    
    with tab2:
        st.subheader("Wajah Pertama")
        image1 = capture_image("camera_face1")
        if image1:
            st.image(image1, caption='Wajah 1', use_column_width=True)
        
            st.subheader("Wajah Kedua")
            image2 = capture_image("camera_face2")
            if image2:
                st.image(image2, caption='Wajah 2', use_column_width=True)
                process_face_similarity(image1, image2)

def process_ethnicity_detection(image):
    model = load_ethnicity_model()
    
    with st.spinner("Mendeteksi dan memproses..."):
        label, confidence = predict_ethnicity(image, model)
        
        if isinstance(label, str) and "tidak" in label:
            st.error(label)
        else:
            st.success(f"Prediksi: **{label}** ({confidence*100:.2f}%)")
            
            # Tampilkan probabilitas untuk semua etnis
            st.subheader("Probabilitas per Etnis:")
            img = image.resize((160, 160))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            probabilities = model.predict(img_array)[0]
            
            chart_data = pd.DataFrame({
                'Etnis': LABELS,
                'Probabilitas': probabilities
            })
            st.bar_chart(chart_data.set_index('Etnis'))
            
            # Tampilkan embedding (opsional)
            with st.expander("🔍 Lihat Embedding (opsional)"):
                embedding = get_face_embedding(image)
                if embedding is not None:
                    st.dataframe(pd.DataFrame(embedding.reshape(1, -1)))
                else:
                    st.write("Embedding tidak tersedia.")

def process_face_similarity(image1, image2):
    with st.spinner("Menghitung kemiripan..."):
        # Dapatkan embedding untuk kedua gambar
        embedding1 = get_face_embedding(image1)
        embedding2 = get_face_embedding(image2)
        
        if embedding1 is None or embedding2 is None:
            st.error("Wajah tidak terdeteksi pada salah satu gambar")
            return
        
        # Hitung kemiripan menggunakan fungsi dari face_similarity_faces.py
        similarity = get_similarity(embedding1, embedding2)
        
        # Tampilkan hasil
        st.subheader("Hasil Perbandingan")
        
        # Visualisasi dengan progress bar
        st.progress(float(similarity))  # Konversi ke float
        
        # Interpretasi hasil
        if similarity >= 0.7:
            st.success(f"Wajah sangat mirip! (Similarity: {similarity:.4f})")
        elif similarity >= 0.5:
            st.info(f"Wajah cukup mirip (Similarity: {similarity:.4f})")
        else:
            st.warning(f"Wajah tidak mirip (Similarity: {similarity:.4f})")
        
        # Tampilkan embedding (opsional)
        with st.expander("🔍 Lihat Embedding (opsional)"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Embedding Wajah 1")
                st.dataframe(pd.DataFrame(embedding1.reshape(1, -1)))
            with col2:
                st.subheader("Embedding Wajah 2")
                st.dataframe(pd.DataFrame(embedding2.reshape(1, -1)))

# ===== MAIN APP =====
def main():
    st.set_page_config(
        page_title="Analisis Wajah",
        page_icon="👤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("Analisis Wajah")
        st.image("https://media.istockphoto.com/id/1299138327/vector/facial-recognition-concept.jpg?s=612x612&w=0&k=20&c=pdPFg6Nx-NKuHKwJP8ljGI97wFPD8XfUsQBHgYwFZVA=", width=200)
        
        # Menu selection
        menu = st.radio(
            "Pilih Menu:",
            ["Beranda", "Deteksi Etnis", "Perbandingan Wajah"]
        )
        
        st.divider()
        st.write("Powered by FaceNet & TensorFlow")
        st.caption("© 2025 Analisis Wajah Indonesia")

    # Main content based on menu selection
    if menu == "Beranda":
        st.title("📊 Selamat Datang di Aplikasi Analisis Wajah")
        
        st.markdown("""
        ### Apa yang bisa dilakukan aplikasi ini?
        
        - **Deteksi Etnis**: Mengidentifikasi etnis berdasarkan gambar wajah
        - **Perbandingan Wajah**: Menghitung tingkat kemiripan antara dua wajah
        
        ### Cara Penggunaan
        
        1. Pilih menu yang diinginkan pada sidebar
        2. Upload gambar atau gunakan kamera
        3. Lihat hasil analisis
        
        ### Catatan
        
        - Aplikasi ini menggunakan model deep learning untuk analisis wajah
        - Akurasi deteksi dapat bervariasi tergantung kualitas gambar
        - Semua gambar yang diunggah hanya diproses secara lokal
        """)
        
        # Contoh gambar
        st.subheader("Contoh Fitur")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://media.istockphoto.com/id/1324786380/vector/face-recognition-icon-face-id-scan-facial-biometric-verification-concept-vector-illustration.jpg?s=612x612&w=0&k=20&c=b9tLPhh5R7oWqhTQsiPpXMQHQYqEASmKZgUUJEHZHcg=", caption="Deteksi Etnis")
        with col2:
            st.image("https://media.istockphoto.com/id/1212228111/vector/face-recognition-system-concept.jpg?s=612x612&w=0&k=20&c=gzUCJGYqbYWjYjYVXKS7J2q5EvpUcd82w9nUBxnvdpQ=", caption="Perbandingan Wajah")
        
    elif menu == "Deteksi Etnis":
        ethnicity_detection_ui()
        
    elif menu == "Perbandingan Wajah":
        face_similarity_ui()

if __name__ == "__main__":
    main()