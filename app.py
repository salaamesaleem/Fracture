import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os

# -----------------------------------------------------------
# 1️⃣  App Title
# -----------------------------------------------------------
st.title("🦴 Fracture Detection using YOLO")
st.write("Upload an X-ray image to detect fractures.")

# -----------------------------------------------------------
# 2️⃣  Load Model
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # your trained model path
    return model

model = load_model()

# -----------------------------------------------------------
# 3️⃣  Image Upload
# -----------------------------------------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image with PIL
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to a temporary file (YOLO expects a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        temp_path = temp.name

    # -------------------------------------------------------
    # 4️⃣  Run Detection
    # -------------------------------------------------------
    st.write("Detecting fractures...")
    results = model(temp_path)

    # results[0].show() won't work in Streamlit — use OpenCV to render boxes
    result_img = results[0].plot()  # returns an annotated numpy image (BGR)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # convert to RGB

    st.image(result_img, caption="Detection Result", use_column_width=True)

    # -------------------------------------------------------
    # 5️⃣  Display details (optional)
    # -------------------------------------------------------
    boxes = results[0].boxes
    if boxes:
        st.subheader("Detections:")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"• {model.names[cls_id]} — Confidence: {conf:.2f}")
    else:
        st.write("No fractures detected.")

    # Clean up temp file
    os.remove(temp_path)
