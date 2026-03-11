import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DermaScan · AI Skin Analysis",
    page_icon="🩺",
    layout="centered"
)

# ─────────────────────────────────────────
# Global CSS  — Warm Clinical Light Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Syne:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ─── Tokens ─── */
:root {
    --bg:       #f5f3ef;
    --surface:  #ffffff;
    --border:   #e4e0d8;
    --green:    #1a7a5e;
    --green-lt: #e6f4f0;
    --blue:     #1c5fa3;
    --blue-lt:  #e8f0fb;
    --red:      #c0392b;
    --red-lt:   #fdecea;
    --text:     #1a1612;
    --muted:    #8a8078;
    --tag-bg:   #eeebe4;
}

/* ─── Layout ─── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
footer { display: none !important; }
.block-container {
    padding: 0 1.5rem 5rem !important;
    max-width: 680px !important;
}

/* ─── Hero ─── */
.hero {
    background: var(--surface);
    border-radius: 20px;
    border: 1px solid var(--border);
    padding: 2.8rem 2rem 2.2rem;
    text-align: center;
    margin: 2rem 0 1.5rem;
    box-shadow: 0 2px 24px rgba(0,0,0,0.05);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; left: 50%;
    transform: translateX(-50%);
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(26,122,94,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--green-lt);
    color: var(--green);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-badge::before { content: '●'; font-size: 0.5rem; animation: blink 2s step-end infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.hero-title {
    font-family: 'Lora', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 0.5rem;
    line-height: 1.15;
    letter-spacing: -0.02em;
}
.hero-title em {
    font-style: italic;
    color: var(--green);
}
.hero-desc {
    font-size: 0.88rem;
    color: var(--muted);
    font-weight: 400;
    line-height: 1.7;
    max-width: 380px;
    margin: 0 auto;
}

/* ─── Workflow Steps ─── */
.workflow {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin: 0 0 1.5rem;
}
.wf-step {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 0.6rem;
    text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.wf-icon { font-size: 1.3rem; display: block; margin-bottom: 0.4rem; }
.wf-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: var(--green);
    letter-spacing: 0.1em;
    display: block;
    margin-bottom: 0.2rem;
}
.wf-label { font-size: 0.72rem; color: var(--muted); font-weight: 500; }

/* ─── Upload Panel ─── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
    transition: border-color 0.25s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--green) !important; }
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 1.6rem 1rem !important;
}
[data-testid="stFileUploaderFileName"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--green) !important;
}

/* ─── Image Preview ─── */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
}

/* ─── Meta Chips ─── */
.chips {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.8rem 0 1.2rem;
    justify-content: center;
}
.chip {
    background: var(--tag-bg);
    border-radius: 100px;
    padding: 0.28rem 0.8rem;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--muted);
}
.chip b { color: var(--text); font-weight: 500; }

/* ─── Result Cards ─── */
.rcard {
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    margin: 1rem 0;
    border-left: 4px solid;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    animation: fadeUp 0.35s ease;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}
.rcard.ok     { background: var(--green-lt); border-color: var(--green); }
.rcard.detect { background: var(--blue-lt);  border-color: var(--blue);  }
.rcard.warn   { background: var(--red-lt);   border-color: var(--red);   }

.rcard-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-radius: 100px;
    padding: 0.22rem 0.65rem;
    margin-bottom: 0.8rem;
}
.rcard.ok     .rcard-tag { background: rgba(26,122,94,0.15);  color: var(--green); }
.rcard.detect .rcard-tag { background: rgba(28,95,163,0.15);  color: var(--blue);  }
.rcard.warn   .rcard-tag { background: rgba(192,57,43,0.15);  color: var(--red);   }

.rcard-title {
    font-family: 'Lora', serif;
    font-size: 1.55rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.3rem;
    line-height: 1.25;
}
.rcard-sub {
    font-size: 0.83rem;
    color: var(--muted);
    line-height: 1.65;
    font-weight: 400;
}

/* ─── Confidence Meter ─── */
.meter { margin-top: 1.2rem; }
.meter-row {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.meter-row b { color: var(--text); }
.meter-bg {
    height: 7px;
    background: rgba(0,0,0,0.08);
    border-radius: 100px;
    overflow: hidden;
}
.meter-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s cubic-bezier(.4,0,.2,1);
}
.rcard.ok     .meter-fill { background: linear-gradient(90deg, #2e9e7b, var(--green)); }
.rcard.detect .meter-fill { background: linear-gradient(90deg, #3580d4, var(--blue));  }

/* ─── Divider ─── */
.sect-divider {
    height: 1px;
    background: var(--border);
    margin: 1.8rem 0;
    border: none;
}

/* ─── Disclaimer ─── */
.disclaimer {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    margin-top: 2rem;
}
.d-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
.d-text { font-size: 0.78rem; color: var(--muted); line-height: 1.65; }
.d-text strong { color: var(--text); font-weight: 600; }

/* ─── Spinner ─── */
[data-testid="stSpinner"] p { color: var(--green) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered · Real-Time</div>
    <h1 class="hero-title">Skin Analysis<br><em>made precise.</em></h1>
    <p class="hero-desc">
        Upload a clear photo of a skin lesion. Our deep-learning model
        will verify, analyse, and return a clinical result in seconds.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Workflow Steps
# ─────────────────────────────────────────
st.markdown("""
<div class="workflow">
    <div class="wf-step">
        <span class="wf-icon">📁</span>
        <span class="wf-num">STEP 01</span>
        <span class="wf-label">Upload</span>
    </div>
    <div class="wf-step">
        <span class="wf-icon">🔍</span>
        <span class="wf-num">STEP 02</span>
        <span class="wf-label">Verify Skin</span>
    </div>
    <div class="wf-step">
        <span class="wf-icon">🧠</span>
        <span class="wf-num">STEP 03</span>
        <span class="wf-label">Analyse</span>
    </div>
    <div class="wf-step">
        <span class="wf-icon">📋</span>
        <span class="wf-num">STEP 04</span>
        <span class="wf-label">Results</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load Model & Class Names
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_disease_model.keras", compile=False)

@st.cache_resource
def load_class_names():
    return np.load("class_names.npy", allow_pickle=True)

model       = load_model()
class_names = load_class_names()

# ─────────────────────────────────────────
# Upload Widget
# ─────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop an image here, or click to browse — JPG · JPEG · PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

# ─────────────────────────────────────────
# Prediction Pipeline
# ─────────────────────────────────────────
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((224, 224))

        # Centered image preview
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(image, use_container_width=True)

        img_array = np.array(image)

        # ── Skin Detection ──
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        skin_mask = np.logical_and.reduce((
            r > 95, g > 40, b > 20,
            (img_array.max(2) - img_array.min(2)) > 15,
            np.abs(r.astype(int) - g.astype(int)) > 15,
            r > g, r > b
        ))
        skin_ratio = skin_mask.sum() / (224 * 224)

        # Meta chips
        fmt = uploaded_file.type.split("/")[-1].upper()
        st.markdown(f"""
        <div class="chips">
            <span class="chip">Resolution&nbsp;<b>224 × 224</b></span>
            <span class="chip">Format&nbsp;<b>{fmt}</b></span>
            <span class="chip">Skin coverage&nbsp;<b>{skin_ratio*100:.1f}%</b></span>
        </div>
        """, unsafe_allow_html=True)

        # ── Skin gate ──
        if skin_ratio < 0.35:
            st.markdown("""
            <div class="rcard warn">
                <div class="rcard-tag">⚠ Verification Failed</div>
                <div class="rcard-title">Not a skin image</div>
                <div class="rcard-sub">
                    Skin coverage is below the required 35% threshold.
                    Please upload a clear, close-up photograph of the affected skin area.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Model inference ──
            img_in = np.expand_dims(img_array / 255.0, axis=0)

            with st.spinner("Running deep-learning analysis…"):
                time.sleep(0.3)
                preds = model.predict(img_in)

            confidence    = float(preds.max())
            predicted_idx = int(preds.argmax())
            disease_name  = class_names[predicted_idx]
            conf_pct      = f"{confidence * 100:.1f}"

            # ── Result card ──
            if confidence < 0.50:
                st.markdown(f"""
                <div class="rcard ok">
                    <div class="rcard-tag">✓ Clear</div>
                    <div class="rcard-title">Normal Skin</div>
                    <div class="rcard-sub">
                        No disease pattern was identified. The image appears to show healthy skin.
                    </div>
                    <div class="meter">
                        <div class="meter-row">
                            <span>Confidence score</span>
                            <b>{conf_pct}%</b>
                        </div>
                        <div class="meter-bg">
                            <div class="meter-fill" style="width:{conf_pct}%"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rcard detect">
                    <div class="rcard-tag">🩺 Condition Detected</div>
                    <div class="rcard-title">{disease_name}</div>
                    <div class="rcard-sub">
                        A skin condition has been identified. Please consult a qualified
                        dermatologist for a professional examination and treatment plan.
                    </div>
                    <div class="meter">
                        <div class="meter-row">
                            <span>Confidence score</span>
                            <b>{conf_pct}%</b>
                        </div>
                        <div class="meter-bg">
                            <div class="meter-fill" style="width:{conf_pct}%"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    except Exception:
        st.markdown("""
        <div class="rcard warn">
            <div class="rcard-tag">⚠ Error</div>
            <div class="rcard-title">Invalid file</div>
            <div class="rcard-sub">
                The file could not be processed. Please upload a valid JPG, JPEG, or PNG image.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# Disclaimer
# ─────────────────────────────────────────
st.markdown("""
<hr class="sect-divider">
<div class="disclaimer">
    <span class="d-icon">⚕️</span>
    <p class="d-text">
        <strong>Medical Disclaimer —</strong>
        DermaScan AI is an assistive tool intended for educational and screening purposes only.
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always seek the guidance of a qualified dermatologist or healthcare provider
        regarding any skin condition.
    </p>
</div>
""", unsafe_allow_html=True)
