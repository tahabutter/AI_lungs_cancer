# AI-Driven Lung Cancer Diagnostic System 🫁🔬

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11+-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end medical imaging solution that leverages **Transfer Learning (MobileNetV2)** to classify lung CT scans into three categories: **Malignant**, **Benign**, and **Normal**. The project includes a high-performance deep learning model, a Flask API, and a modern web dashboard.

## 🚀 Project Overview
Early detection of lung cancer significantly increases survival rates. This project provides an automated screening tool designed to assist radiologists by providing a second opinion with high clinical precision.

### Key Features:
- **Transfer Learning Integration:** Utilizes MobileNetV2 for robust feature extraction.
- **Dynamic Localization:** Automatically identifies and boxes "Effected Areas" in positive scans.
- **Advanced Preprocessing:** Uses CLAHE and Gaussian Blur for superior image enhancement.
- **Professional UI:** A modern, glassmorphism-themed web dashboard for real-time analysis.

## 📊 Model Performance
The model was validated on the **IQ-OTHNCCD Lung Cancer Dataset** and achieved state-of-the-art results:

- **Overall Accuracy:** 97%
- **Malignant Recall:** 100% (Zero False Negatives)
- **F1-Score:** 0.97 (Weighted Average)

### Classification Report:
| Class            | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Benign Cases     | 0.79      | 0.96   | 0.87     |
| Malignant Cases  | 1.00      | 1.00   | 1.00     |
| Normal Cases     | 0.99      | 0.93   | 0.96     |


## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow, Keras, MobileNetV2
- **Computer Vision:** OpenCV, Matplotlib, Seaborn
- **Backend:** Flask, Flask-CORS
- **Frontend:** HTML5, Tailwind CSS, JavaScript (ES6+)

## 🔍 Pipeline Architecture

### 1. Data Preprocessing
To improve the visibility of nodules, the following pipeline was implemented:
- **Resizing:** Standardized to 224x224 pixels.
- **CLAHE:** Applied Contrast Limited Adaptive Histogram Equalization to sharpen lung textures.
- **Gaussian Blur:** Removed high-frequency noise from CT scans.

### 2. Model Training
We transitioned from a custom CNN to **Transfer Learning** to improve generalization.
- **Base Model:** MobileNetV2 (Pre-trained on ImageNet).
- **Optimizer:** Adam with a Dynamic Learning Rate Scheduler (`ReduceLROnPlateau`).
- **Callbacks:** Early Stopping to prevent overfitting.

## 💻 Installation & Usage

### 1. Clone the repository
git clone [https://github.com/tahabutter/AI_lungs_cancer.git](https://github.com/tahabutter/AI_lungs_cancer.git) 

2. Install Dependencies

pip install -r requirements.txt

3. Run the API

python app.py

4. Launch the Dashboard

Open index.html in your favorite browser. Upload a CT scan and click "Run AI Analysis".

📂 Project Structure

├── lung_dataset/           # Dataset directory
├── static/                 # CSS and JS assets
├── app.py                  # Flask API Backend
├── lung_ai_mobilenet.py    # Training Script
├── index.html              # Frontend Dashboard
├── requirements.txt        # Python Dependencies
└── lung_cancer_mobilenet_v2.keras # Saved Model

🤝 Contributing

Contributions are welcome! If you'd like to improve the detection logic or UI, please fork the repo and create a pull request.

👤 Author
Muhammad Taha

LinkedIn: www.linkedin.com/in/muhammad taha-4a5b2a293

University: Baba Guru Nanak University, Nankana Sahib

cd AI_lungs_cancer
