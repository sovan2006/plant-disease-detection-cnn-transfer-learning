import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# =====================================================
# Flask App
# =====================================================

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================================
# Model Path
# =====================================================

MODEL_PATH = "best_plant_disease_model.keras"

# =====================================================
# Load Model
# =====================================================

try:
    model = load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False,
        safe_mode=False
    )

    print("✅ Model Loaded Successfully")

except Exception as e:
    print("❌ Error Loading Model")
    print(e)
    model = None

# =====================================================
# Class Names
# Replace these with YOUR dataset class names
# =====================================================

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___healthy",
    "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight",
    "Grape___healthy",
    "Orange___Haunglongbing",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper___Bacterial_spot",
    "Pepper___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Mosaic_virus",
    "Tomato___healthy"
]

# =====================================================
# Recommendations
# =====================================================

recommendation_map = {
    "Tomato___Late_blight":
        "Apply copper fungicide and remove infected leaves.",

    "Tomato___healthy":
        "Plant is healthy. No treatment required.",

    "Potato___Early_blight":
        "Use Mancozeb fungicide and avoid overhead irrigation.",

    "Apple___Apple_scab":
        "Apply fungicide and prune infected branches."
}

# =====================================================
# Prediction Function
# =====================================================

def predict_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    class_index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    disease = class_names[class_index]

    recommendation = recommendation_map.get(
        disease,
        "No recommendation available."
    )

    return (
        disease,
        f"{confidence*100:.2f}%",
        recommendation
    )

# =====================================================
# Home
# =====================================================

@app.route("/")
def index():
    return render_template("index.html")

# =====================================================
# Predict
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:

        return render_template(
            "index.html",
            disease="Model Not Loaded",
            confidence="--",
            recommendation="Please check terminal for loading errors."
        )

    if "file" not in request.files:

        return render_template(
            "index.html",
            disease="No Image",
            confidence="--",
            recommendation="Upload an image."
        )

    file = request.files["file"]

    if file.filename == "":

        return render_template(
            "index.html",
            disease="No Image",
            confidence="--",
            recommendation="Please choose an image."
        )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    disease, confidence, recommendation = predict_image(filepath)

    return render_template(
        "index.html",
        disease=disease,
        confidence=confidence,
        recommendation=recommendation,
        image_path="/" + filepath.replace("\\", "/")
    )

# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)
    
    