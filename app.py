import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

MODEL_PATH = Path("outputs/models/brain_tumor_classifier.keras")
CLASS_NAMES_PATH = Path("outputs/class_names.json")


# ---------------------------------------------------------
# Load the trained model only once
# ---------------------------------------------------------

@st.cache_resource
def load_saved_model():
    return load_model(MODEL_PATH)


# ---------------------------------------------------------
# Load class names
# ---------------------------------------------------------

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# Prepare the uploaded MRI image
# ---------------------------------------------------------

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    return np.expand_dims(image_array, axis=0)


# ---------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Image Classification")

st.write(
    "Upload a brain MRI image to obtain a model prediction."
)

st.warning(
    "For educational and research purposes only. "
    "This application does not provide a medical diagnosis."
)


# ---------------------------------------------------------
# Check that the required files exist
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not CLASS_NAMES_PATH.exists():
    st.error(f"Class names file not found: {CLASS_NAMES_PATH}")
    st.stop()


# ---------------------------------------------------------
# Load saved model and class names
# ---------------------------------------------------------

model = load_saved_model()
class_names = load_class_names()


# ---------------------------------------------------------
# Upload an MRI image
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    if st.button("Predict"):
        with st.spinner("Analysing image..."):
            image_array = preprocess_image(image)
            probabilities = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_class = class_names[predicted_index]
            confidence = float(probabilities[predicted_index])

        st.subheader("Prediction Result")
        st.write(f"**Predicted class:** {predicted_class}")
        st.write(f"**Model confidence:** {confidence:.2%}")

        st.subheader("Class Scores")
        probability_data = {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }

        st.bar_chart(probability_data)