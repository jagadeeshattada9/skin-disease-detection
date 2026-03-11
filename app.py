import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🩺",
    layout="centered"
)

# ----------------------------------------------------
# CSS Styling
# ----------------------------------------------------
st.markdown("""
<style>

body{
background:#f4f6f9;
font-family:'Segoe UI',sans-serif;
}

.block-container{
max-width:720px;
padding-top:20px;
}

/* HERO */

.hero{
text-align:center;
padding:35px;
background:linear-gradient(135deg,#1abc9c,#3498db);
border-radius:18px;
color:white;
margin-bottom:25px;
}

.hero h1{
font-size:36px;
margin-bottom:10px;
}

.hero p{
font-size:16px;
opacity:0.9;
}

/* WORKFLOW */

.workflow{
display:grid;
grid-template-columns:repeat(4,1fr);
gap:10px;
margin-bottom:25px;
}

.step{
background:white;
border-radius:12px;
padding:12px;
text-align:center;
border:1px solid #e1e4e8;
font-size:14px;
box-shadow:0 2px 6px rgba(0,0,0,0.05);
}

.step span{
font-size:22px;
display:block;
}

/* IMAGE */

.center-img img{
border-radius:15px;
box-shadow:0 6px 20px rgba(0,0,0,0.15);
margin-top:10px;
}

/* RESULT */

.result-box{
padding:20px;
border-radius:14px;
margin-top:25px;
font-size:16px;
}

.success{
background:#e8f8f5;
border-left:6px solid #1abc9c;
}

.warning{
background:#fdecea;
border-left:6px solid #e74c3c;
}

.detect{
background:#ebf5fb;
border-left:6px solid #3498db;
}

/* UPLOADER */

[data-testid="stFileUploader"]{
background:white;
border-radius:12px;
border:2px dashed #3498db;
padding:15px;
}

/* FOOTER */

.footer{
margin-top:40px;
padding:18px;
background:white;
border-radius:12px;
border:1px solid #e1e4e8;
font-size:13px;
color:#666;
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Hero Section
# ----------------------------------------------------
st.markdown("""
<div class="hero">
<h1>🩺 DermaScan AI</h1>
<p>Upload a skin image and our AI model will analyze it for possible skin conditions.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Steps Section
# ----------------------------------------------------
st.markdown("""
<div class="workflow">

<div class="step">
<span>📤</span>
Upload Image
</div>

<div class="step">
<span>🔍</span>
Verify Skin
</div>

<div class="step">
<span>🧠</span>
Analyze
</div>

<div class="step">
<span>📋</span>
Results
</div>

</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_disease_model.keras", compile=False)

@st.cache_resource
def load_classes():
    return np.load("class_names.npy", allow_pickle=True)

model = load_model()
class_names = load_classes()

# ----------------------------------------------------
# Upload Image
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg","jpeg","png"]
)

# ----------------------------------------------------
# Prediction Pipeline
# ----------------------------------------------------
if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224,224))
        img_array = np.array(image)

        # Show Image
        st.markdown("<div class='center-img'>", unsafe_allow_html=True)
        st.image(image,width=350)
        st.markdown("</div>", unsafe_allow_html=True)

        # Skin Detection
        r,g,b = img_array[:,:,0],img_array[:,:,1],img_array[:,:,2]

        skin_mask = np.logical_and.reduce((
            r > 95,
            g > 40,
            b > 20,
            (img_array.max(2)-img_array.min(2)) > 15,
            np.abs(r.astype(int)-g.astype(int)) > 15,
            r > g,
            r > b
        ))

        skin_ratio = skin_mask.sum()/(224*224)

        # Skin Verification
        if skin_ratio < 0.30:

            st.markdown("""
            <div class="result-box warning">
            <h3>⚠ Not a Skin Image</h3>
            <p>Please upload a clear close-up image of skin.</p>
            </div>
            """, unsafe_allow_html=True)

        else:

            img_input = np.expand_dims(img_array/255.0, axis=0)

            with st.spinner("Running AI analysis..."):
                time.sleep(0.5)
                preds = model.predict(img_input)

            confidence = float(preds.max())
            predicted_idx = int(preds.argmax())
            disease_name = class_names[predicted_idx]

            conf_percent = confidence * 100

            # Reduced confidence threshold
            if confidence < 0.30:

                st.markdown(f"""
                <div class="result-box success">
                <h3>✅ Healthy Skin</h3>
                <p>No disease pattern detected.</p>
                <b>Confidence: {conf_percent:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="result-box detect">
                <h3>🩺 Condition Detected</h3>
                <p><b>{disease_name}</b></p>
                <p>Please consult a dermatologist for professional advice.</p>
                <b>Confidence: {conf_percent:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)

    except:
        st.error("Invalid image file")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("""
<div class="footer">
⚕️ <b>Medical Disclaimer:</b> This AI tool is for educational purposes only and does not replace professional medical diagnosis.
</div>
""", unsafe_allow_html=True)
