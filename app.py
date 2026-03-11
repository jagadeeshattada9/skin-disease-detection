import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="centered"
)

# ---------------------------------
# Custom CSS - Clean Clinical Dark Theme
# ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root & Reset ── */
:root {
    --bg:        #0a0d12;
    --surface:   #111520;
    --border:    #1e2535;
    --accent:    #3ee8b5;
    --accent2:   #4f8ef7;
    --warn:      #f7774f;
    --text:      #dce6f0;
    --muted:     #5a6a80;
    --card:      #141926;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 720px !important; }
footer { visibility: hidden; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2.5rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(62,232,181,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 1rem;
    filter: drop-shadow(0 0 18px rgba(62,232,181,0.5));
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { filter: drop-shadow(0 0 14px rgba(62,232,181,0.4)); }
    50%       { filter: drop-shadow(0 0 28px rgba(62,232,181,0.8)); }
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    color: #fff;
    margin: 0 0 0.4rem;
    line-height: 1.1;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}
.hero-divider {
    width: 60px; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 1.5rem auto 0;
    border: none;
}

/* ── Step Badges ── */
.steps-row {
    display: flex;
    gap: 0;
    margin: 2rem 0 1.5rem;
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    background: var(--surface);
}
.step {
    flex: 1;
    padding: 1rem 0.8rem;
    text-align: center;
    border-right: 1px solid var(--border);
    position: relative;
}
.step:last-child { border-right: none; }
.step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.3rem;
}
.step-label {
    font-size: 0.78rem;
    color: var(--muted);
    font-weight: 500;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    transition: border-color 0.3s;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 2rem 1rem !important;
}
[data-testid="stFileUploadDropzone"] > div {
    color: var(--muted) !important;
    font-size: 0.9rem !important;
}
.uploadedFileName { color: var(--accent) !important; font-family: 'DM Mono', monospace; font-size: 0.8rem !important; }

/* ── Image Display ── */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5) !important;
}

/* ── Result Cards ── */
.result-card {
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin: 1.2rem 0;
    border: 1px solid;
    position: relative;
    overflow: hidden;
    animation: slideUp 0.4s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: inherit;
    opacity: 0.06;
}
.result-card.success {
    background: rgba(62,232,181,0.06);
    border-color: rgba(62,232,181,0.3);
}
.result-card.disease {
    background: rgba(79,142,247,0.06);
    border-color: rgba(79,142,247,0.3);
}
.result-card.error {
    background: rgba(247,119,79,0.06);
    border-color: rgba(247,119,79,0.3);
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-label.success { color: var(--accent); }
.result-label.disease { color: var(--accent2); }
.result-label.error   { color: var(--warn); }
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    color: #fff;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}
.result-meta {
    font-size: 0.85rem;
    color: var(--muted);
    font-weight: 300;
}

/* ── Confidence Bar ── */
.conf-wrap { margin-top: 1.2rem; }
.conf-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
    font-family: 'DM Mono', monospace;
}
.conf-track {
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}
.conf-fill.success { background: linear-gradient(90deg, #2ec99a, var(--accent)); }
.conf-fill.disease { background: linear-gradient(90deg, #3a72e8, var(--accent2)); }

/* ── Skin Ratio Pill ── */
.meta-pills {
    display: flex;
    gap: 0.6rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    color: var(--muted);
}
.pill span { color: var(--text); }

/* ── Disclaimer Banner ── */
.disclaimer {
    margin-top: 2.5rem;
    padding: 1rem 1.2rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
}
.disclaimer-icon { font-size: 1rem; margin-top: 1px; flex-shrink: 0; }
.disclaimer-text { font-size: 0.78rem; color: var(--muted); line-height: 1.6; }
.disclaimer-text strong { color: var(--text); }

/* ── Spinner Override ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ── Streamlit default overrides ── */
h1, h2, h3 { color: #fff !important; }
p { color: var(--text); }
[data-testid="stMarkdownContainer"] p { color: var(--text); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Hero Header
# ---------------------------------
st.markdown("""
<div class="hero">
    <span class="hero-icon">🔬</span>
    <h1 class="hero-title">Derma<span>Scan</span> AI</h1>
    <p class="hero-sub">Clinical Skin Lesion Analysis · Powered by Deep Learning</p>
    <hr class="hero-divider">
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# Step Indicators
# ---------------------------------
st.markdown("""
<div class="steps-row">
    <div class="step">
        <span class="step-num">01</span>
        <span class="step-label">Upload Image</span>
    </div>
    <div class="step">
        <span class="step-num">02</span>
        <span class="step-label">Skin Verification</span>
    </div>
    <div class="step">
        <span class="step-num">03</span>
        <span class="step-label">AI Analysis</span>
    </div>
    <div class="step">
        <span class="step-num">04</span>
        <span class="step-label">Results</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Model (Cached)
# ---------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "skin_disease_model.keras",
        compile=False
    )

@st.cache_resource
def load_class_names():
    return np.load("class_names.npy", allow_pickle=True)

model = load_model()
class_names = load_class_names()

# ---------------------------------
# File Upload
# ---------------------------------
uploaded_file = st.file_uploader(
    "Drop your skin lesion image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

# ---------------------------------
# Prediction Section
# ---------------------------------
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224, 224))

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="", use_container_width=True)

        img_array = np.array(image)

        # ---------------------------------
        # SKIN DETECTION FILTER
        # ---------------------------------
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]

        skin_pixels = np.logical_and.reduce((
            r > 95, g > 40, b > 20,
            (np.max(img_array, axis=2) - np.min(img_array, axis=2)) > 15,
            np.abs(r - g) > 15,
            r > g, r > b
        ))

        skin_ratio = np.sum(skin_pixels) / (224 * 224)

        # Show image meta pills
        st.markdown(f"""
        <div class="meta-pills">
            <div class="pill">Resolution <span>224 × 224 px</span></div>
            <div class="pill">Skin coverage <span>{skin_ratio*100:.1f}%</span></div>
            <div class="pill">Format <span>{uploaded_file.type.split('/')[1].upper()}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Reject non-skin images
        if skin_ratio < 0.35:
            st.markdown("""
            <div class="result-card error">
                <div class="result-label error">⚠ Verification Failed</div>
                <div class="result-title">Not a Skin Image</div>
                <div class="result-meta">Insufficient skin coverage detected. Please upload a clear, close-up photograph of a skin lesion or affected area.</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ---------------------------------
            # MODEL PREDICTION
            # ---------------------------------
            img_normalized = img_array / 255.0
            img_input = np.expand_dims(img_normalized, axis=0)

            with st.spinner("Running deep neural network analysis…"):
                time.sleep(0.4)  # Small pause for UX
                prediction = model.predict(img_input)

            confidence = float(np.max(prediction))
            predicted_class = int(np.argmax(prediction))
            disease_name = class_names[predicted_class]

            # ---------------------------------
            # CONFIDENCE THRESHOLD
            # ---------------------------------
            if confidence < 0.50:
                conf_pct = f"{confidence*100:.1f}"
                st.markdown(f"""
                <div class="result-card success">
                    <div class="result-label success">✓ Analysis Complete</div>
                    <div class="result-title">Normal Skin Detected</div>
                    <div class="result-meta">No significant skin disease pattern was identified in this image.</div>
                    <div class="conf-wrap">
                        <div class="conf-header">
                            <span>Model Confidence</span>
                            <span>{conf_pct}%</span>
                        </div>
                        <div class="conf-track">
                            <div class="conf-fill success" style="width:{conf_pct}%"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                conf_pct = f"{confidence*100:.1f}"
                st.markdown(f"""
                <div class="result-card disease">
                    <div class="result-label disease">🩺 Condition Identified</div>
                    <div class="result-title">{disease_name}</div>
                    <div class="result-meta">A skin condition was detected. Please consult a dermatologist for professional evaluation.</div>
                    <div class="conf-wrap">
                        <div class="conf-header">
                            <span>Model Confidence</span>
                            <span>{conf_pct}%</span>
                        </div>
                        <div class="conf-track">
                            <div class="conf-fill disease" style="width:{conf_pct}%"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown("""
        <div class="result-card error">
            <div class="result-label error">⚠ Processing Error</div>
            <div class="result-title">Invalid Image File</div>
            <div class="result-meta">Unable to process the uploaded file. Please ensure you're uploading a valid JPG, JPEG, or PNG image.</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------
# Medical Disclaimer
# ---------------------------------
st.markdown("""
<div class="disclaimer">
    <span class="disclaimer-icon">⚕️</span>
    <span class="disclaimer-text">
        <strong>Medical Disclaimer:</strong> DermaScan AI is an assistive tool for educational purposes only.
        Results are not a substitute for professional medical diagnosis or treatment.
        Always consult a qualified dermatologist or healthcare provider for any skin concerns.
    </span>
</div>
""", unsafe_allow_html=True)
