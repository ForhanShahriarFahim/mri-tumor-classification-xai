import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(page_title="MRI Tumor Classifier & XAI", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .main {background-color: #f4f6f9;}
    h1 {color: #1e3a8a; font-family: 'Helvetica Neue', sans-serif;}
    .stButton>button {
        background-color: #2563eb; color: white; border-radius: 8px;
        padding: 10px 24px; font-weight: bold; border: none; transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {background-color: #1d4ed8; color: white;}
    .metric-container {
        background-color: white; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Grad-CAM Implementation
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to extract feature maps and gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class_index):
        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        # Target specific class and backward pass
        target = output[0][target_class_index]
        target.backward()

        # Global average pooling on gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach().clone()
        
        # Weight activations by gradients
        for i in range(activations.shape[1]):
            activations[0, i, :, :] *= pooled_gradients[i]
            
        # Average and apply ReLU
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        # Normalize between 0 and 1
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
            
        return heatmap.cpu().numpy()

def overlay_heatmap(heatmap, original_image, alpha=0.5):
    """Overlays the Grad-CAM heatmap onto the original image."""
    if not isinstance(original_image, np.ndarray):
        original_image = np.array(original_image)
        
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend images
    overlayed_image = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed_image

# ==========================================
# 3. Model Loading & Preprocessing
# ==========================================
# Classes must be in exact alphabetical order matching the Kaggle dataset folders
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize ResNet50 architecture
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4) # 4 classes for your dataset
    
    # Load the trained weights
    model_path = 'best_mri_model.pth'
    if not os.path.exists(model_path):
        st.error(f"Error: '{model_path}' not found. Please ensure your trained model file is in the same folder as app.py.")
        st.stop()
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Preprocessing pipeline (must match your training setup)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 4. Streamlit UI Layout
# ==========================================
st.title("🧠 MRI Brain Tumor Classification & XAI")
st.markdown("""
This application classifies MRI brain scans into four categories: **Glioma, Meningioma, Pituitary Tumor, or No Tumor**. 
It utilizes a PyTorch ResNet50 model and employs **Grad-CAM** to highlight the regions of the brain that influenced the model's decision, building trust through explainable AI.
""")
st.divider()

# Sidebar for uploading
with st.sidebar:
    st.header("Upload MRI Scan")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.info("Supported formats: JPG, JPEG, PNG")
    st.markdown("---")
    st.markdown("**Note:** For the best results, upload a clear, cropped axial, coronal, or sagittal MRI scan.")

# Main working area
if uploaded_file is not None:
    # Read and display original image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Original MRI Scan")
        st.image(image, use_container_width=True)
        
    # Analyze Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Analyze MRI Scan"):
        with st.spinner('Analyzing scan and generating activation maps...'):
            
            # 1. Preprocess the image
            input_tensor = preprocess(image).unsqueeze(0).to(device)
            
            # 2. Make Prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            predicted_class = CLASSES[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            # 3. Generate Grad-CAM Explanation
            # Target the last convolutional block of ResNet50
            target_layer = model.layer4[-1] 
            cam = GradCAM(model, target_layer)
            
            # We must enable gradients temporarily to generate Grad-CAM
            with torch.enable_grad():
                heatmap = cam.generate(input_tensor, predicted_idx.item())
                
            # 4. Create Overlay Image
            
            original_resized = image.resize((224, 224))
            result_image = overlay_heatmap(heatmap, original_resized)
            
            # 5. Display Results
            with col2:
                st.subheader("Grad-CAM Explanation")
                st.image(result_image, use_container_width=True)
            
            st.divider()
            
            # Display Final Metrics
            st.markdown(f"""
            <div class="metric-container">
                <h2 style="color: #1e3a8a; margin-bottom: 0;">Diagnosis: {predicted_class}</h2>
                <h4 style="color: #64748b; margin-top: 5px;">Confidence Score: {confidence_score:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            
else:
    st.info("👈 Please upload an MRI scan from the sidebar to begin the analysis.")