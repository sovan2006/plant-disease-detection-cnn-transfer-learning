import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(
    page_title="AI Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

MODEL_PATH = "best_plant_disease_model.keras"

CLASS_NAMES = [
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

RECOMMENDATION_MAP = {
    "Tomato___Late_blight":
        "Apply an appropriate copper-based fungicide and remove severely infected leaves.",

    "Tomato___healthy":
        "The prediction indicates a healthy tomato leaf. Continue normal crop care and monitoring.",

    "Potato___Early_blight":
        "Consider an appropriate fungicide and avoid prolonged leaf wetness.",

    "Apple___Apple_scab":
        "Consider suitable fungicide management and remove infected plant material."
}


@st.cache_resource(show_spinner="Loading AI model...")
def load_ai_model():
    return load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )


def predict_image(uploaded_image, model):

    img = uploaded_image.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.asarray(
        img,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    prediction = model.predict(
        img_array,
        verbose=0
    )

    class_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction)
    )

    disease = CLASS_NAMES[class_index]

    recommendation = RECOMMENDATION_MAP.get(
        disease,
        "No specific recommendation is configured for this class."
    )

    return disease, confidence, recommendation


# =========================
# UI
# =========================

st.title("🌿 AI Plant Disease Detection")

st.markdown(
    """
    ### CNN + EfficientNetB0 Transfer Learning

    Upload a plant leaf image and let the AI model
    identify the predicted disease.
    """
)

st.divider()


# Load model
try:

    model = load_ai_model()

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.code(str(e))

    st.stop()


# Upload image
uploaded_file = st.file_uploader(
    "📤 Upload Plant Leaf Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Predict Disease",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing leaf image..."
        ):

            disease, confidence, recommendation = predict_image(
                image,
                model
            )

        disease_name = (
            disease
            .replace("___", " — ")
            .replace("_", " ")
        )

        st.success(
            f"Prediction: {disease_name}"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Model",
                "EfficientNetB0"
            )

        st.progress(
            confidence
        )

        st.subheader(
            "🌱 Recommendation"
        )

        st.info(
            recommendation
        )


st.divider()

st.caption(
    "Powered by TensorFlow + EfficientNetB0 | "
    "PlantVillage Dataset"
)
