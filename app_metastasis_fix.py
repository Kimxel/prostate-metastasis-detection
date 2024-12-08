import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import time
import torch.nn as nn
from timm import create_model

# Cache the model to avoid reloading on every app refresh
@st.cache_resource
def load_model():
    try:
        # Create and modify the DeiT model for 6 classes
        model = create_model('deit_small_patch16_224', pretrained=True)
        model.head = nn.Linear(model.head.in_features, 6)
        if hasattr(model, 'head_dist'):
            model.head_dist = nn.Linear(model.head_dist.in_features, 6)

        # Load the state dictionary (model weights)
        model.load_state_dict(torch.load(
            "models/best_model-deit1.pth",
            map_location=torch.device('cpu')
        ))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

model = load_model()

# Define label mapping
label_map = {0: 'M0', 1: 'M1', 2: 'M1a', 3: 'M1b', 4: 'M1c', 5: 'Mx'}

# Image preprocessing
def preprocess_image(image):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None

# Function to classify an image
def classify_image(image):
    processed_image = preprocess_image(image)
    if processed_image is not None:
        output = model(processed_image)
        _, predicted_class = torch.max(output, 1)
        predicted_label = label_map[predicted_class.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0, predicted_class].item()
        return predicted_label, confidence
    else:
        return None, None

import pandas as pd  # Untuk tabel data

# UI: Folder (multiple files) upload

# App UI
st.title("Prostate Cancer Metastasis Detection")
# st.header("Upload an MRI Image for Classification")
st.markdown("""
Unggah gambar MRI untuk memprediksi stadium metastasis kanker prostat. Model akan mengklasifikasikan gambar ke dalam salah satu kategori berikut:
- **M0**: Tidak ada tanda metastasis atau penyebaran kanker
- **M1**: Kanker telah menyebar ke bagian tubuh lain di luar panggul (pelvis)
- **M1a**: Sel kanker terdapat pada kelenjar getah bening di luar panggul (pelvis)
- **M1b**: Sel kanker terdapat di dalam tulang
- **M1c**: Sel kanker terdapat di bagian tubuh lain seperti paru-paru
- **Mx**: Status metastasis tidak diketahui
""")

import time  # Untuk mencatat waktu
import pandas as pd  # Untuk tabel data

# UI: Folder (multiple files) upload
folder_files = st.file_uploader("Pilih sebuah file gambar atau banyak gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if folder_files:
    try:
        start_time = time.time()  # Catat waktu mulai
        results = []

        for uploaded_file in folder_files:
            image = Image.open(uploaded_file).convert("RGB")
            with st.spinner(f"Classifying {uploaded_file.name}..."):
                predicted_label, confidence = classify_image(image)
                results.append({
                    "Image": image,  # Simpan gambar
                    "File Name": uploaded_file.name,
                    "Predicted Class": predicted_label if predicted_label else "Error",
                    "Confidence (%)": f"{confidence * 100:.2f}" if confidence else "N/A"
                })

        end_time = time.time()  # Catat waktu selesai
        total_time = end_time - start_time  # Hitung total waktu yang dihabiskan

        # Display results with images
        st.success("Klasifikasi Berhasil!")
        st.write(f"**Total waktu klasifikasi**: {total_time:.2f} seconds")
        st.write("Berikut adalah hasil prediksi:")

        # Custom result display
        for result in results:
            col1, col2 = st.columns([1, 3])  # Layout: gambar di kiri, teks di kanan
            with col1:
                st.image(result["Image"], caption=result["File Name"], width=150)  # Tampilkan gambar
            with col2:
                st.write(f"**File Name**: {result['File Name']}")
                st.write(f"**Predicted Class**: {result['Predicted Class']}")
                st.write(f"**Confidence**: {result['Confidence (%)']}%")
            st.write("---")  # Garis pemisah antar hasil
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
