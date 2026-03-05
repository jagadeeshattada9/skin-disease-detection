import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Skin Disease Detection System")
st.write("Upload a skin lesion image to detect the disease.")

# ---------------------------------
# Load Model (Cached)
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
# File Upload
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
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224, 224))

        st.image(image, caption="Uploaded Image", width=300)

        img_array = np.array(image)

        # ---------------------------------
        # SKIN DETECTION FILTER
        # ---------------------------------
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]

        skin_pixels = np.logical_and.reduce((
            r > 95,
            g > 40,
            b > 20,
            (np.max(img_array, axis=2) - np.min(img_array, axis=2)) > 15,
            np.abs(r - g) > 15,
            r > g,
            r > b
        ))

        skin_ratio = np.sum(skin_pixels) / (224 * 224)

        # Reject non-skin images
        if skin_ratio < 0.35:
            st.error("❌ This does not appear to be a skin image. Please upload a proper skin lesion image.")

        else:
            # ---------------------------------
            # MODEL PREDICTION
            # ---------------------------------
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            with st.spinner("Analyzing image..."):
                prediction = model.predict(img_array)

            confidence = np.max(prediction)
            predicted_class = np.argmax(prediction)
            disease_name = class_names[predicted_class]

            # ---------------------------------
            # CONFIDENCE THRESHOLD
            # ---------------------------------
            if confidence < 0.50:
                st.success("✅ No skin disease detected (Normal Skin)")
                st.info(f"Model Confidence: {confidence*100:.2f}%")
            else:
                st.success(f"🩺 Predicted Disease: {disease_name}")
                st.info(f"Confidence: {confidence*100:.2f}%")

    except Exception as e:
        st.error("⚠️ Error processing image. Please upload a valid image file.")
