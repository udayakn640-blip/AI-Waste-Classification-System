"""Streamlit interface for the trained waste image classifier."""
from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("model/waste_classifier.keras")
LABELS_PATH = Path("model/class_names.json")
IMAGE_SIZE = (160,  160)
CONFIDENCE_WARNING = 0.60

st.set_page_config(page_title="AI Waste Classifier", page_icon="♻️", layout="centered")
st.title("♻️ AI-Based Waste Classification System")
st.caption("AI/ML internship prototype • Supports SDG 12: Responsible Consumption and Production")

st.write(
    "Upload a clear photo containing one main waste item. The model predicts one of "
    "the categories it was trained on."
)

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.warning(
        "The trained model is not available yet. First prepare the labelled dataset "
        "and run `python train_model.py`. Follow the instructions in README.md, then "
        "restart this app."
    )
    st.stop()

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, labels

try:
    model, class_names = load_model()
except Exception as exc:
    st.error(f"Could not load the model. Check the installation and model files. Details: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Choose a waste-item image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        resized = image.resize(IMAGE_SIZE)
        arr = np.asarray(resized, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        probabilities = model.predict(arr, verbose=0)[0]
        best_idx = int(np.argmax(probabilities))
        label = class_names[best_idx]
        confidence = float(probabilities[best_idx])

        st.subheader("Prediction")
        st.success(f"Predicted category: **{label.replace('_', ' ').title()}**")
        st.metric("Model confidence (estimate)", f"{confidence:.1%}")

        tips = {
            "paper": "Keep paper clean and dry where local recycling rules accept it.",
            "plastic": "Check the resin/type and local collection rules; not all plastics are accepted.",
            "glass": "Follow local glass collection rules and handle broken glass carefully.",
            "metal": "Check local rules for cans and other metal items.",
            "organic": "Food and garden scraps may be suitable for composting where available.",
        }
        key = label.lower().replace(" ", "_")
        st.info(tips.get(key, "Check your local waste-management guidance for this category."))

        if confidence < CONFIDENCE_WARNING:
            st.warning(
                "The model is not very confident. Treat this result as uncertain and verify "
                "the item manually."
            )
        st.caption(
            "This is an educational prototype. Predictions can be wrong, and local waste "
            "rules differ. Do not rely on it to identify hazardous, medical, or electronic waste."
        )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
