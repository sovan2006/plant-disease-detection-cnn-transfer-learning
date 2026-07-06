# 🌿 AI Powered Plant Disease Detection
### CNN + Transfer Learning (EfficientNetB0) | TensorFlow | Flask | Railway

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-2.17-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras">
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/Railway-Deployed-success?style=for-the-badge">
</p>

---

# 📌 Project Overview

This project is an **AI-powered web application** that detects plant diseases from leaf images using **Deep Learning**.

The application leverages **Transfer Learning with EfficientNetB0** to classify plant diseases with high accuracy and provides instant predictions through a clean web interface.

The goal is to assist farmers, researchers, and agriculture enthusiasts in identifying plant diseases quickly without requiring expert knowledge.

---

# 🚀 Live Demo

**Live Application**

link: https://plant-disease-detection-cnn-transfer-learning-production.up.railway.app/

---

# 📷 Project Preview

## Home Page

<img width="1440" height="744" alt="Screenshot 2026-07-07 at 3 04 08 AM" src="https://github.com/user-attachments/assets/66bc6de1-1a7d-42b7-840e-cdc9b82d47ae" />

## Prediction Result

<img width="1432" height="773" alt="Screenshot 2026-07-07 at 3 04 17 AM" src="https://github.com/user-attachments/assets/746c795d-c347-4aed-b71d-8ccd77a23236" />



---

# ✨ Features

✅ Upload plant leaf image

✅ AI-based disease prediction

✅ Confidence score

✅ Disease identification

✅ Responsive UI

✅ Fast inference

✅ Flask backend

✅ Railway Deployment

---

# 🧠 Deep Learning Pipeline

```
Leaf Image
      │
      ▼
Image Preprocessing
      │
      ▼
Resize (224 × 224)
      │
      ▼
Normalization
      │
      ▼
EfficientNetB0
(Transfer Learning)
      │
      ▼
Dense Layers
      │
      ▼
Softmax Classification
      │
      ▼
Disease Prediction
```

---

# 🏗️ Project Architecture

```
                User
                  │
                  ▼
          Upload Leaf Image
                  │
                  ▼
             Flask Backend
                  │
                  ▼
      Image Preprocessing
                  │
                  ▼
 EfficientNetB0 Deep Learning Model
                  │
                  ▼
 Disease Prediction + Confidence
                  │
                  ▼
       HTML/CSS/JavaScript UI
```

---

# 📂 Project Structure

```
Plant-Disease-Detection/
│
├── static/
│   ├── uploads/
│   ├── css/
│   └── images/
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── runtime.txt
├── Procfile
├── best_plant_disease_model.keras
├── README.md
└── plantdiseasedetection.ipynb
```

---

# 🛠 Tech Stack

## Programming

- Python

## Deep Learning

- TensorFlow
- Keras
- EfficientNetB0

## Computer Vision

- Pillow
- NumPy

## Backend

- Flask

## Frontend

- HTML5
- CSS3
- JavaScript

## Deployment

- Railway
- Gunicorn

---

# 📊 Model Information

| Model | Accuracy |
|---------|----------|
| CNN | 92.4% |
| EfficientNetB0 | **95.7%** |

---

# 📚 Dataset

**PlantVillage Dataset**

- 54,305 Images
- 38 Classes

Contains healthy and diseased leaf images from multiple plant species.

---

# Supported Diseases

Examples include:

- Apple Scab
- Apple Black Rot
- Cedar Apple Rust
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Potato Early Blight
- Potato Late Blight
- Pepper Bacterial Spot
- Corn Rust
- Grape Black Rot

and many more...

---

# ⚙️ Installation

Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detection-cnn-transfer-learning.git
```

Move inside folder

```bash
cd plant-disease-detection-cnn-transfer-learning
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 🧪 Model Workflow

```
Dataset

↓

EDA

↓

Image Preprocessing

↓

Data Augmentation

↓

Transfer Learning

↓

Model Training

↓

Fine Tuning

↓

Evaluation

↓

Save Model (.keras)

↓

Flask Deployment

↓

Railway Deployment
```

---

# 📈 Future Improvements

- Mobile Application

- Disease Recommendation Engine

- Fertilizer Suggestion

- Treatment Recommendation

- Multi-language Support

- Camera Capture

- Explainable AI (Grad-CAM)

- Cloud Database

---

# 💼 Skills Demonstrated

✔ Deep Learning

✔ Transfer Learning

✔ TensorFlow

✔ CNN

✔ Computer Vision

✔ Flask Development

✔ REST Architecture

✔ Model Deployment

✔ Railway Cloud

✔ Git & GitHub

✔ Machine Learning Pipeline

---

# 👨‍💻 Author

**Sovan Barik**

B.Tech Artificial Intelligence & Machine Learning

Passionate about:

- Machine Learning
- Deep Learning
- Computer Vision
- Generative AI
- MLOps

GitHub:

https://github.com/sovan2006

LinkedIn:

https://www.linkedin.com/in/sovan-barik-711bba326/
---

# ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.

It motivates me to build more AI projects.
