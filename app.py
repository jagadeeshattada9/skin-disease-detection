import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ─────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🩺",
    layout="centered"
)

# ─────────────────────────────────────────
# Custom CSS (Clean + Professional)
# ─────────────────────────────────────────
st.markdown("""
<style>

body{
    background-color:#f5f3ef;
}

.block-container{
    max-width:720px;
    padding-top:1rem;
}

.hero{
    text-align:center;
    padding:30px;
    background:white;
    border-radius:16px;
    border:1px solid #e5e2da;
    margin-bottom:25px;
}

.hero h1{
    font-size:36px;
    margin-bottom:10px;
}

.hero p{
    color:#777;
    font-size:15px;
}

.result-box{
    padding:20px;
    border-radius:12px;
    margin-top:20px;
}

.success{
    background:#e8f5f0;
    border-left:6px solid #1a7a5e;
}

.warning{
    background:#fdecea;
    border-left:6px solid #c0392b;
}

.detect{
    background:#e8f0fb;
    border-left:6px solid #1c5fa3;
}

.center-img{
    text-align:center;
}

.metric-box{
    background:white;
    border-radius:10px;
    padding:8px 12px;
    border:1px solid #e5e2da;
    text-align:center;
}

.footer{
    margin-top:40px;
    padding:15px;
    background:white;
    border-radius:10px;
    border:1px solid #e5e2da;
    font-size:13px;
    color:#777;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Hero Section
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
<h1>🩺 DermaScan AI</h1>
<p>Upload a skin image and our AI model will analyse it for possible skin conditions.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_disease_model.keras", compile=False)

@st.cache_resource
def load_class_names():
    return np.load("class_names.npy", allow_pickle=True)

model = load_model()
class_names = load_class_names()


# ─────────────────────────────────────────
# Skin Detection Function
# ─────────────────────────────────────────
def detect_skin_ratio(image_array):

    r,g,b = image_array[:,:,0],image_array[:,:,1],image_array[:,:,2]

    mask = np.logical_and.reduce((

        r > 95,
        g > 40,
        b > 20,
        (image_array.max(2)-image_array.min(2)) > 15,
        np.abs(r.astype(int)-g.astype(int)) > 15,
        r > g,
        r > b
    ))

    ratio = mask.sum() / (224*224)

    return ratio


# ─────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────
def predict_disease(image_array):

    img = np.expand_dims(image_array/255.0,axis=0)

    preds = model.predict(img)

    confidence = float(preds.max())

    idx = int(preds.argmax())

    disease = class_names[idx]

    return disease,confidence


# ─────────────────────────────────────────
# Image Upload
# ─────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg","jpeg","png"]
)


# ─────────────────────────────────────────
# Processing Pipeline
# ─────────────────────────────────────────
if uploaded_file:

    try:

        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224,224))
        img_array = np.array(image)

        st.markdown("<div class='center-img'>",unsafe_allow_html=True)
        st.image(image,width=300)
        st.markdown("</div>",unsafe_allow_html=True)


        # ── Metadata
        col1,col2,col3 = st.columns(3)

        with col1:
            st.markdown("<div class='metric-box'>Resolution<br><b>224×224</b></div>",unsafe_allow_html=True)

        with col2:
            fmt = uploaded_file.type.split("/")[-1].upper()
            st.markdown(f"<div class='metric-box'>Format<br><b>{fmt}</b></div>",unsafe_allow_html=True)

        # Skin detection
        skin_ratio = detect_skin_ratio(img_array)

        with col3:
            st.markdown(f"<div class='metric-box'>Skin %<br><b>{skin_ratio*100:.1f}%</b></div>",unsafe_allow_html=True)


        # ── Skin Gate
        if skin_ratio < 0.35:

            st.markdown("""
            <div class="result-box warning">
            <h3>⚠ Not a Skin Image</h3>
            <p>Please upload a clear close-up image of skin.</p>
            </div>
            """,unsafe_allow_html=True)


        else:

            with st.spinner("Running AI analysis..."):
                time.sleep(0.5)

                disease,confidence = predict_disease(img_array)

            conf_percent = confidence*100

            if confidence < 0.50:

                st.markdown(f"""
                <div class="result-box success">
                <h3>✅ Healthy Skin</h3>
                <p>No disease pattern detected.</p>
                <b>Confidence: {conf_percent:.1f}%</b>
                </div>
                """,unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="result-box detect">
                <h3>🩺 Condition Detected</h3>
                <p><b>{disease}</b></p>
                <p>Please consult a dermatologist for medical advice.</p>
                <b>Confidence: {conf_percent:.1f}%</b>
                </div>
                """,unsafe_allow_html=True)


    except:

        st.error("Invalid image file")


# ─────────────────────────────────────────
# Footer / Disclaimer
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
⚕️ <b>Medical Disclaimer:</b> This AI tool is for educational and screening purposes only. 
It does not replace professional medical diagnosis. Always consult a dermatologist.
</div>
""",unsafe_allow_html=True)
