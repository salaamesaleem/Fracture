import streamlit as st
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import os
import gdown
from PIL import Image

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------
st.set_page_config(page_title="Fracture Detection", layout="wide")
st.title(" Fracture Detection App (YOLO-based)")

# --------------------------------------------------
# MODEL LOADING (Safe & Cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    import torch
    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import DetectionModel
    import gdown
    import os

    model_path = "best.pt"
    url = "https://drive.google.com/file/d/1yF2j8_d_V27wI7oxfOUD5pxq1sQJOXqQ/view?usp=sharing"  # <-- put your own model ID

    # Download model if not already present
    if not os.path.exists(model_path):
        with st.spinner("Downloading YOLO model... please wait ⏳"):
            gdown.download(url, model_path, quiet=False)

    # Allow YOLO model class to be safely unpickled
    add_safe_globals([DetectionModel])

    # Load YOLO model
    model = YOLO(model_path)
    return model



try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --------------------------------------------------
# IMAGE UPLOAD SECTION
# --------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload an X-ray image for fracture detection", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert PIL → OpenCV format
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run YOLO prediction
    with st.spinner("🔍 Detecting fractures..."):
        results = model.predict(source=img_bgr, conf=0.5, imgsz=640)

    # Get first prediction result
    res_plotted = results[0].plot()  # Draw bounding boxes
    st.image(res_plotted, caption="Detection Result", use_container_width=True)

    # Show detected classes and confidence
    detected = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls] if hasattr(model, 'names') else f"Class {cls}"
        detected.append((label, conf))

    if detected:
        st.subheader("🧠 Detected Fractures:")
        for label, conf in detected:
            st.write(f"• {label} ({conf*100:.1f}% confidence)")
    else:
        st.warning("No fractures detected. ✅")

else:
    st.info("Please upload an image to start fracture detection.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
---
👨‍⚕️ **Fracture Detection App** — powered by YOLOv8  
Developed for demonstration purposes.
""")
