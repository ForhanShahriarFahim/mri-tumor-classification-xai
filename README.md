# 🧠 MRI Brain Tumor Classification & Explainable AI (XAI)

An end-to-end deep learning web application that classifies brain MRI scans into four categories: **Glioma, Meningioma, Pituitary Tumor, or No Tumor**. 

Bridging the gap between advanced computer vision and practical application, this tool utilizes a fine-tuned PyTorch ResNet50 model. Crucially, it addresses the "black box" nature of deep learning by implementing **Grad-CAM (Gradient-weighted Class Activation Mapping)**. By visualizing neuron activation patterns, the app provides transparent, visual explanations for its predictions, helping to build trust in AI-assisted medical analysis.

## 📸 Application Interface

*(Below are screenshots of the application in action)*

### 1. Upload & Prediction
![Classification UI](Image/Brain%20MRI%20Classification%20Img-1.png)

### 2. Explainable AI (Grad-CAM) Overlay
![Grad-CAM Results](Image/Brain%20Mri%20Classification%20Image-2.png)

## ✨ Key Features
* **Multi-Class Classification:** Accurately distinguishes between three types of brain tumors and healthy brain scans.
* **Explainable AI:** Custom PyTorch hooks extract gradients from the final convolutional layer to generate heatmaps, showing exactly *where* the model is looking.
* **Interactive UI:** A clean, user-friendly interface built with Streamlit, designed for ease of use by non-technical users.
* **Robust Preprocessing:** Handles image resizing, normalization (ImageNet standards), and tensor conversion seamlessly in the background.

## 📊 Dataset
This model was trained on the curated **Brain Tumor MRI Dataset** (Version 2), comprising 7,200 human brain MRI images. The dataset features strictly balanced classes and eliminated overlap between training and testing sets to prevent data leakage.

## 🚀 How to Run Locally

### Prerequisites
* Python 3.8+
* Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
