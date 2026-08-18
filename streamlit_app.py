import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_PATH = "best_plant_disease_model.keras"


# =========================================================
# CLASS NAMES
# =========================================================

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


# =========================================================
# DISEASE RECOMMENDATIONS
# =========================================================

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


# =========================================================
# LOAD AI MODEL
# =========================================================

@st.cache_resource(show_spinner="Loading AI model...")
def load_ai_model():

    return load_model(
        MODEL_PATH,

        # IMPORTANT:
        # The saved model contains preprocess_input
        # inside a Lambda layer.
        custom_objects={
            "preprocess_input": preprocess_input
        },

        compile=False,

        # Required because the saved model contains
        # a Lambda layer.
        safe_mode=False
    )


# =========================================================
# IMAGE PREDICTION
# =========================================================

def predict_image(uploaded_image, model):

    # Convert image to RGB
    img = uploaded_image.convert("RGB")

    # Resize to EfficientNetB0 input size
    img = img.resize((224, 224))

    # Convert image to NumPy array
    img_array = np.asarray(
        img,
        dtype=np.float32
    )

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # EfficientNet preprocessing
    img_array = preprocess_input(
        img_array
    )

    # Model prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    # Get predicted class
    class_index = int(
        np.argmax(prediction)
    )

    # Get confidence
    confidence = float(
        np.max(prediction)
    )

    # Get disease name
    disease = CLASS_NAMES[class_index]

    # Get recommendation
    recommendation = RECOMMENDATION_MAP.get(
        disease,
        "No specific recommendation is configured for this class."
    )

    return (
        disease,
        confidence,
        recommendation
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "🌿 AI Plant Disease Detection"
)

st.markdown(
    """
    ### CNN + EfficientNetB0 Transfer Learning

    Upload a plant leaf image and let the AI model
    identify the predicted disease.
    """
)

st.divider()


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_ai_model()

except Exception as e:

    st.error(
        "❌ Model could not be loaded."
    )

    st.error(
        "Please check the model file and TensorFlow/Keras version."
    )

    st.code(
        str(e),
        language="text"
    )

    st.stop()


# =========================================================
# IMAGE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload Plant Leaf Image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],

    help="Upload a clear image of a plant leaf."
)


# =========================================================
# IMAGE PROCESSING
# =========================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    st.write("")

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if st.button(
        "🔍 Predict Disease",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "🌱 Analyzing leaf image..."
        ):

            try:

                disease, confidence, recommendation = predict_image(
                    image,
                    model
                )

            except Exception as e:

                st.error(
                    "❌ Prediction failed."
                )

                st.code(
                    str(e),
                    language="text"
                )

                st.stop()


        # =================================================
        # FORMAT DISEASE NAME
        # =================================================

        disease_name = (
            disease
            .replace(
                "___",
                " — "
            )
            .replace(
                "_",
                " "
            )
        )


        # =================================================
        # RESULT
        # =================================================

        st.success(
            f"🌿 Prediction: {disease_name}"
        )


        # =================================================
        # METRICS
        # =================================================

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


        # =================================================
        # CONFIDENCE BAR
        # =================================================

        st.write(
            "Prediction Confidence"
        )

        st.progress(
            min(
                max(
                    confidence,
                    0.0
                ),
                1.0
            )
        )


        # =================================================
        # RECOMMENDATION
        # =================================================

        st.subheader(
            "🌱 Recommendation"
        )

        st.info(
            recommendation
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Powered by TensorFlow + EfficientNetB0 | "
    "PlantVillage Dataset"
)

st.caption(
    "⚠️ Predictions are for educational/demo purposes "
    "and should not replace professional agricultural advice."
)
