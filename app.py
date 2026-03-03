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
# Load Class Names (List format)
# ---------------------------------
class_names = np.load("class_names.npy", allow_pickle=True)

# ---------------------------------
# Upload Image
# ---------------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# Prediction
# ---------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    st.image(image, caption="Uploaded Image", width=350)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    disease_name = class_names[predicted_class]

    st.success(f"🩺 Predicted Disease: {disease_name}")