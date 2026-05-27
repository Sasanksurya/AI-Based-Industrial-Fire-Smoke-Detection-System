import streamlit as st
import cv2
import os
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="🔥 Industrial AI Monitoring",
    page_icon="🔥",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🔥 Industrial AI Monitoring")
st.sidebar.markdown("---")

# =========================
# TITLE
# =========================

st.title("🔥 AI-Based Industrial Fire & Smoke Detection System")

st.markdown("""
Real-time industrial monitoring dashboard powered by:
- YOLOv8
- OpenCV
- Streamlit
- Telegram Alerts
- Email Alerts
""")

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# =========================
# STATUS BOX
# =========================

st.success("✅ System Running Successfully")

# =========================
# INCIDENT DIRECTORY
# =========================

SAVE_DIR = "incidents"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# =========================
# DASHBOARD LAYOUT
# =========================

col1, col2 = st.columns([1, 3])

# =========================
# SIDEBAR CONTROLS
# =========================

with col1:

    st.subheader("🎛 Monitoring Controls")

    start_button = st.button("▶ Start Monitoring")
    stop_button = st.button("⏹ Stop Monitoring")

    st.markdown("---")

    incident_images = [
        img for img in os.listdir(SAVE_DIR)
        if img.endswith((".jpg", ".png", ".jpeg"))
    ]

    st.metric("📸 Total Incidents", len(incident_images))

# =========================
# MAIN AREA
# =========================

with col2:

    st.subheader("📹 Live Monitoring")

    frame_placeholder = st.empty()

# =========================
# WEBCAM FUNCTION
# =========================

run = False

if start_button:
    run = True

if stop_button:
    run = False

# =========================
# LOCAL WEBCAM NOTE
# =========================

st.warning("""
⚠ Streamlit Cloud cannot access your local webcam.

Use this dashboard locally for live monitoring.

Cloud deployment is mainly for:
- Dashboard preview
- Incident viewing
- Portfolio showcase
""")

# =========================
# INCIDENT GALLERY
# =========================

st.markdown("---")
st.subheader("📂 Incident Gallery")

incident_images = sorted(
    incident_images,
    reverse=True
)

if len(incident_images) > 0:

    cols = st.columns(3)

    for idx, img_name in enumerate(incident_images[:9]):

        img_path = os.path.join(SAVE_DIR, img_name)

        with cols[idx % 3]:
            st.image(
                img_path,
                caption=img_name,
                use_container_width=True
            )

else:
    st.info("No incidents recorded yet.")

# =========================
# FOOTER
# =========================

st.markdown("---")
st.markdown("### 👨‍💻 Developed by Shashank Surya")