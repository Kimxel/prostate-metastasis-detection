import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import torch.nn as nn
from timm import create_model

# Cache the model to avoid reloading on every app refresh
@st.cache_resource
def load_model():
    try:
        # Create and modify the DeiT model for 6 classes
        model = create_model('deit_small_patch16_224', pretrained=True)  # Load the pretrained model
        model.head = nn.Linear(model.head.in_features, 6)  # Adjust for 6-class classification

        # If the DeiT model has a distilled head, modify it as well
        if hasattr(model, 'head_dist'):
            model.head_dist = nn.Linear(model.head_dist.in_features, 6)

        # Load the state dictionary (model weights)
        model.load_state_dict(torch.load(
            "E:\\Kimi\\Penelitian\\Transformer_Skripsi\\Deit\\Deit_Biasa_Tuning\\best_model-deit1.pth",
            map_location=torch.device('cpu')
        ))
        model.eval()  # Set the model to evaluation mode
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
            transforms.Resize((224, 224)),  # Resize to input size expected by DeiT
            transforms.ToTensor(),          # Convert PIL image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ImageNet standards
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return None

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

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB if grayscale
        # st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Display uploaded image with a fixed width
        st.image(image, caption="Uploaded Image", width=300)
        st.write("")

        # Preprocess image
        processed_image = preprocess_image(image)

        if processed_image is not None:
            # Model inference with timing
            with st.spinner("Classifying..."):
                start_time = time.time()  # Record start time
                output = model(processed_image)
                end_time = time.time()  # Record end time

                _, predicted_class = torch.max(output, 1)
                predicted_label = label_map[predicted_class.item()]
                confidence = torch.nn.functional.softmax(output, dim=1)[0, predicted_class].item()

            # Compute and display time taken
            time_taken = end_time - start_time

            # Display results
            st.success(f"Predicted Class: {predicted_label}")
            st.write(f"Confidence: {confidence * 100:.2f}%")
            st.write(f"Time taken for prediction: {time_taken:.2f} seconds")
        else:
            st.error("Failed to preprocess the image. Please try again.")
    except Exception as e:
        st.error(f"An error occurred during classification: {str(e)}")
else:
    st.info("Please upload an image to begin the classification process.")
