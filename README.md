# Prostate Cancer Metastasis Detection Website

This Streamlit application classifies MRI images of the prostate into six metastasis stages using a Vision Transformer model variant, which is Data-efficient Image Transformer.

## Features
- Upload an MRI image to predict the metastasis stage of prostate cancer.
- Real-time classification with confidence scores and processing time displayed.
- Supports six metastasis categories:
  - **M0**: No signs of metastasis.
  - **M1**: Cancer has spread beyond the pelvis.
  - **M1a**: Cancer in lymph nodes outside the pelvis.
  - **M1b**: Cancer in bones.
  - **M1c**: Cancer in other organs, e.g., lungs.
  - **Mx**: Metastasis status unknown.

## How It Works
- The app loads a fine-tuned **DeiT model** (`deit_small_patch16_224`) trained for 6-class classification.
- Input images are resized and preprocessed to match the model's expected format.
- The model predicts the metastasis stage and provides the confidence level for each prediction.

## Deployment
The app is deployed on Streamlit Cloud. You can access it directly at: https://prostate-metastasis-detection.streamlit.app/

## Model Information
The application uses a Vision Transformer (DeiT) model fine-tuned for prostate cancer metastasis detection. The model weights are stored in the repository under models/best_model-deit1.pth.

## Training Details
- Model architecture: DeiT (Data-Efficient Image Transformer)
- Pretrained on: ImageNet
- Fine-tuned on: Custom prostate MRI dataset with six metastasis labels.

## File Structure
Prostate-Cancer-Detection/
├── app.py                # Main application file
├── models/
│   └── best_model-deit1.pth  # Model weights
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

## Dependencies
The app uses the following libraries:
- Streamlit: Web application framework
- Torch: PyTorch for model inference
- timm: Pretrained Vision Transformer models
- Pillow: Image preprocessing

Thanks 😊
