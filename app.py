import streamlit as st
from ultralytics import YOLO
import gdown
import os
from PIL import Image
import numpy as np
import cv2

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?id=1yF2j8_d_V27wI7oxfOUD5pxq1sQJOXqQ"  # Direct download link

# -------------------------------
# DOWNLOAD MODEL IF NOT EXISTS
# -------------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading YOLOv8 model... ⏳"):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -------------------------------
# LOAD YOLO MODEL
# -------------------------------
try:
    model = YOLO(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("Fracture Detection App (PyTorch .pt model)")

uploaded_file = st.file_uploader("📤 Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert PIL -> OpenCV
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Run YOLO prediction
    with st.spinner("🔍 Detecting fractures..."):
        results = model.predict(source=img_bgr, conf=0.5, imgsz=640)

    # Plot results
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Detection Result", use_container_width=True)

    # Show detected classes & confidence
    detected = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls] if hasattr(model, "names") else f"Class {cls}"
        detected.append((label, conf))

    if detected:
        st.subheader("🧠 Detected Fractures:")
        for label, conf in detected:
            st.write(f"• {label} ({conf*100:.1f}% confidence)")
    else:
        st.warning("No fractures detected. ✅")
else:
    st.info("Please upload an X-ray image to start fracture detection.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
---
👨‍⚕️ **Fracture Detection App** — powered by YOLOv8  
Developed for demonstration purposes.
""")
