import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Skin Disease Detection System")
st.write("Upload a skin lesion image to detect the disease.")

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "skin_disease_model.keras",
        compile=False
    )

model = load_model()

# ---------------------------------
# Load Class Names
# ---------------------------------
class_names = np.load("class_names.npy", allow_pickle=True)

# ---------------------------------
# Upload Image
# ---------------------------------
uploaded_file = st.file_uploader(
    "Choose a skin lesion image...",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# Prediction Section
# ---------------------------------
if uploaded_file is not None:

    try:
        # Open and preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224, 224))

        st.image(image, caption="Uploaded Image", width=300)

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model Prediction
        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)

        # Confidence Threshold
        THRESHOLD = 0.75  # Adjust between 0.70 - 0.85 if needed

        if confidence < THRESHOLD:
            st.error("❌ This does not appear to be a valid skin lesion image. Please upload a proper skin disease image.")
        else:
            disease_name = class_names[predicted_class]
            st.success(f"🩺 Predicted Disease: {disease_name}")
            st.info(f"Confidence: {confidence*100:.2f}%")

    except Exception as e:
        st.error("⚠️ Error processing image. Please upload a valid image file.")
