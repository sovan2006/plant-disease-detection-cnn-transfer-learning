# 🌿 AI Powered Plant Disease Detection

### CNN + Transfer Learning (EfficientNetB0) | TensorFlow | Keras | Streamlit | Computer Vision

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-2.17-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras">
  <img src="https://img.shields.io/badge/EfficientNetB0-Transfer%20Learning-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Computer%20Vision-AI-green?style=for-the-badge">
</p>

---

## 🚀 Live Demo

### 🌐 Streamlit Application

**[🔗 Open Live Application](https://plant-disease-detection-cnn-transfer-learning-bnkxa6nbfr2gsrtj.streamlit.app/)**

Upload a plant leaf image and receive an AI-powered disease prediction with a confidence score and recommendation.

---

# 📌 Project Overview

**AI Powered Plant Disease Detection** is a deep learning web application that identifies plant diseases from leaf images using **Convolutional Neural Networks (CNNs)** and **Transfer Learning with EfficientNetB0**.

The application provides a simple interface where users can upload a plant leaf image and receive:

* 🌿 Predicted disease
* 📊 Prediction confidence
* 🤖 EfficientNetB0 model information
* 🌱 Basic recommendation for selected diseases

The project demonstrates a complete machine learning workflow, from dataset preparation and image preprocessing to model training, evaluation, model saving, and cloud deployment.

> ⚠️ **Disclaimer:** Predictions are intended for educational and demonstration purposes and should not replace professional agricultural advice.

---

# ✨ Features

* 📤 Upload plant leaf images
* 🤖 AI-based disease classification
* 🧠 CNN + EfficientNetB0 transfer learning
* 📊 Prediction confidence score
* 🌱 Disease identification
* 💡 Disease recommendations for configured classes
* 🖥️ Interactive Streamlit interface
* ⚡ Fast inference after model loading
* ☁️ Streamlit Community Cloud deployment
* 📱 Responsive web interface
* 🔄 38-class plant disease classification

---

# 🧠 Deep Learning Pipeline

```text
                 Leaf Image
                     │
                     ▼
               Image Upload
                     │
                     ▼
              RGB Conversion
                     │
                     ▼
             Resize 224 × 224
                     │
                     ▼
       EfficientNet Preprocessing
                     │
                     ▼
              EfficientNetB0
           Transfer Learning
                     │
                     ▼
           Classification Layers
                     │
                     ▼
           Softmax Classification
                     │
                     ▼
          Disease Prediction
                     │
             ┌───────┴───────┐
             ▼               ▼
        Confidence      Recommendation
             │               │
             └───────┬───────┘
                     ▼
              Streamlit UI
```

---

# 🏗️ Application Architecture

```text
                       User
                        │
                        ▼
               Streamlit Web App
                        │
                        ▼
                Upload Leaf Image
                        │
                        ▼
                Image Preprocessing
                        │
                        ▼
             EfficientNetB0 Model
                        │
                        ▼
                Disease Prediction
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       Confidence Score     Recommendation
              │                   │
              └─────────┬─────────┘
                        ▼
                 Streamlit UI
```

---

# 📊 Model Performance

| Model          |  Accuracy |
| -------------- | --------: |
| CNN            |     92.4% |
| EfficientNetB0 | **95.7%** |

### 🏆 Best Model

**EfficientNetB0 — 95.7% reported accuracy**

The EfficientNetB0 transfer learning approach performed better than the baseline CNN model in this project.

---

# 📚 Dataset

## PlantVillage Dataset

The model was trained using the **PlantVillage Dataset**.

### Dataset Information

| Property     | Details                          |
| ------------ | -------------------------------- |
| Dataset      | PlantVillage                     |
| Images       | **54,305+**                      |
| Classes      | **38**                           |
| Task         | Plant Disease Classification     |
| Input        | Plant Leaf Images                |
| Problem Type | Multi-Class Image Classification |

The dataset contains healthy and diseased leaf images from multiple plant species.

---

# 🌱 Supported Disease Classes

The model supports **38 classes**, including healthy and diseased leaves.

Examples:

### 🍎 Apple

* Apple Scab
* Apple Black Rot
* Cedar Apple Rust
* Apple Healthy

### 🍅 Tomato

* Tomato Bacterial Spot
* Tomato Early Blight
* Tomato Late Blight
* Tomato Leaf Mold
* Tomato Septoria Leaf Spot
* Tomato Spider Mites
* Tomato Target Spot
* Tomato Yellow Leaf Curl Virus
* Tomato Mosaic Virus
* Tomato Healthy

### 🥔 Potato

* Potato Early Blight
* Potato Late Blight
* Potato Healthy

### 🌽 Corn

* Corn Cercospora Leaf Spot / Gray Leaf Spot
* Corn Common Rust
* Corn Northern Leaf Blight
* Corn Healthy

### 🍇 Grape

* Grape Black Rot
* Grape Esca / Black Measles
* Grape Leaf Blight
* Grape Healthy

### 🌶️ Pepper

* Pepper Bacterial Spot
* Pepper Healthy

### 🍓 Strawberry

* Strawberry Leaf Scorch
* Strawberry Healthy

And additional classes covering:

* Blueberry
* Cherry
* Orange
* Peach
* Raspberry
* Soybean
* Squash

---

# 🧪 Model Development Workflow

```text
                 PlantVillage Dataset
                         │
                         ▼
                 Dataset Exploration
                         │
                         ▼
                Image Preprocessing
                         │
                         ▼
                  Data Augmentation
                         │
                         ▼
                    CNN Baseline
                         │
                         ▼
                 Transfer Learning
                         │
                         ▼
                   EfficientNetB0
                         │
                         ▼
                   Model Training
                         │
                         ▼
                     Fine Tuning
                         │
                         ▼
                     Evaluation
                         │
                         ▼
                 Save Model (.keras)
                         │
                         ▼
                Streamlit Application
                         │
                         ▼
             Streamlit Cloud Deployment
```

---

# 🔬 Prediction Workflow

```text
User Uploads Image
        ↓
Convert Image to RGB
        ↓
Resize to 224 × 224
        ↓
EfficientNet Preprocessing
        ↓
EfficientNetB0 Inference
        ↓
Class Probabilities
        ↓
Predicted Class
        ↓
Confidence Score
        ↓
Disease Recommendation
        ↓
Display Result
```

---

# 🛠️ Tech Stack

## Programming

* Python 3.11

## Deep Learning

* TensorFlow 2.17
* Keras
* EfficientNetB0
* CNN
* Transfer Learning

## Computer Vision

* Pillow
* NumPy

## Web Application

* Streamlit

## Model Deployment

* Streamlit Community Cloud

## Development & Version Control

* Jupyter Notebook
* Git
* GitHub

---

# 📂 Project Structure

```text
plant-disease-detection-cnn-transfer-learning/
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── streamlit_app.py
│
├── app.py
│
├── best_plant_disease_model.keras
│
├── plantdiseasedetection.ipynb
│
├── requirements.txt
│
├── .python-version
│
├── runtime.txt
│
├── Procfile
│
├── render.yaml
│
└── README.md
```

### Important Files

| File                             | Purpose                            |
| -------------------------------- | ---------------------------------- |
| `streamlit_app.py`               | Main Streamlit web application     |
| `best_plant_disease_model.keras` | Trained EfficientNetB0 model       |
| `plantdiseasedetection.ipynb`    | Model training and experimentation |
| `requirements.txt`               | Python dependencies                |
| `.python-version`                | Python runtime configuration       |
| `app.py`                         | Previous Flask implementation      |
| `README.md`                      | Project documentation              |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/sovan2006/plant-disease-detection-cnn-transfer-learning.git
```

## 2. Enter the Project Directory

```bash
cd plant-disease-detection-cnn-transfer-learning
```

## 3. Create a Virtual Environment

### macOS / Linux

```bash
python3.11 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
streamlit>=1.60,<2
tensorflow==2.17.1
numpy<2
Pillow>=10,<12
h5py>=3.10,<4
```

---

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# ☁️ Deployment

The current production application is deployed using **Streamlit Community Cloud**.

### Deployment Configuration

```text
Repository:
sovan2006/plant-disease-detection-cnn-transfer-learning

Branch:
main

Entry Point:
streamlit_app.py

Python:
3.11

TensorFlow:
2.17.1
```

### 🌐 Live Application

```text
https://plant-disease-detection-cnn-transfer-learning-bnkxa6nbfr2gsrtj.streamlit.app/
```

---

# 🧠 Why EfficientNetB0?

EfficientNetB0 was selected as the transfer learning backbone because it provides a strong balance between:

* Model accuracy
* Computational efficiency
* Parameter efficiency
* Image classification performance
* Practical inference speed

Instead of training a deep CNN completely from scratch, the project uses **transfer learning** to leverage features learned from large-scale image datasets.

---

# 🔄 Transfer Learning Approach

```text
Pretrained EfficientNetB0
          │
          ▼
Remove / Adapt Original Classifier
          │
          ▼
Add Custom Classification Layers
          │
          ▼
Train on PlantVillage Dataset
          │
          ▼
Fine Tune
          │
          ▼
Final Plant Disease Classifier
```

---

# 📈 Future Improvements

The project can be extended with:

### 📱 Mobile Application

Develop an Android/iOS application for field-level plant disease detection.

### 📷 Camera Capture

Allow users to directly capture leaf images using their device camera.

### 💊 Treatment Recommendation

Provide more detailed disease treatment information.

### 🧪 Fertilizer Recommendation

Recommend suitable fertilizer based on crop and disease condition.

### 🌍 Multi-Language Support

Support regional languages such as:

* Bengali
* Hindi
* English

### 🔍 Explainable AI

Implement **Grad-CAM** to visualize which regions of the leaf influenced the prediction.

### ☁️ Cloud Database

Store prediction history and user activity.

### 📊 Analytics Dashboard

Add:

* Prediction history
* Disease frequency
* Crop statistics
* Confidence trends

### 🗣️ Voice Assistant

Add voice-based interaction for farmers.

---

# ⚠️ Limitations

* The model is trained on the PlantVillage dataset and may perform differently on real-world field images.
* Image quality, lighting, background, and leaf orientation can affect predictions.
* Predictions should not be treated as professional agricultural diagnosis.
* Disease recommendations are currently available only for selected classes.
* Real-world deployment would require additional validation on diverse field conditions.

---

# 💼 Skills Demonstrated

### Machine Learning

* Machine Learning Pipeline
* Model Training
* Model Evaluation
* Transfer Learning
* Multi-Class Classification

### Deep Learning

* CNN
* EfficientNetB0
* TensorFlow
* Keras
* Data Augmentation

### Computer Vision

* Image Preprocessing
* Image Resizing
* Image Classification
* Leaf Disease Detection

### Application Development

* Python
* Streamlit
* Interactive Web Applications

### Deployment

* Streamlit Community Cloud
* Git
* GitHub
* Model Deployment

---

# 🎯 Project Highlights

| Category                  | Details                   |
| ------------------------- | ------------------------- |
| 🎯 Problem                | Plant Disease Detection   |
| 🧠 Approach               | CNN + Transfer Learning   |
| 🚀 Backbone               | EfficientNetB0            |
| 📚 Dataset                | PlantVillage              |
| 🖼️ Images                | 54,305+                   |
| 🔢 Classes                | 38                        |
| 📊 Best Reported Accuracy | **95.7%**                 |
| 🐍 Python                 | 3.11                      |
| 🔥 Framework              | TensorFlow / Keras        |
| 🖥️ Web App               | Streamlit                 |
| ☁️ Deployment             | Streamlit Community Cloud |
| 📦 Model                  | `.keras`                  |

---

# 👨‍💻 Author

## Sovan Barik

**B.Tech — Artificial Intelligence & Machine Learning**

Passionate about:

* 🤖 Machine Learning
* 🧠 Deep Learning
* 👁️ Computer Vision
* ✨ Generative AI
* ⚙️ MLOps

### GitHub

[https://github.com/sovan2006](https://github.com/sovan2006)

### LinkedIn

[https://www.linkedin.com/in/sovan-barik-711bba326/](https://www.linkedin.com/in/sovan-barik-711bba326/)

---

# ⭐ Support the Project

If you found this project useful, please consider giving the repository a ⭐ on GitHub.

Your support motivates me to continue building and sharing AI/ML projects.

---

## 🌿 Final Result

**AI Powered Plant Disease Detection**

> Upload → Analyze → Predict → Understand

**EfficientNetB0 + TensorFlow + Streamlit + PlantVillage**

### 🚀 [Try the Live Application](https://plant-disease-detection-cnn-transfer-learning-bnkxa6nbfr2gsrtj.streamlit.app/)
