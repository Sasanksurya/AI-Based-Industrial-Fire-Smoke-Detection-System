import streamlit as st
import cv2
import os
import time
from ultralytics import YOLO

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="🔥 Industrial Fire Monitoring",
    page_icon="🔥",
    layout="wide"
)

# =========================
# LOAD YOLO MODEL
# =========================

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# INCIDENT DIRECTORY
# =========================

SAVE_DIR = "incidents"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# =========================
# TITLE
# =========================

st.title("🔥 AI-Based Industrial Fire & Smoke Detection System")

st.markdown("""
### Real-Time Industrial Monitoring Dashboard

Features:
- Fire Detection
- Smoke Detection
- YOLOv8 AI Model
- Incident Monitoring
- Streamlit Dashboard
""")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🎛 Monitoring Controls")

start_monitoring = st.sidebar.button("▶ Start Monitoring")
stop_monitoring = st.sidebar.button("⏹ Stop Monitoring")

st.sidebar.markdown("---")

# =========================
# STATUS
# =========================

status_box = st.empty()

# =========================
# INCIDENT COUNT
# =========================

incident_images = [
    img for img in os.listdir(SAVE_DIR)
    if img.endswith((".jpg", ".png", ".jpeg"))
]

st.metric("📸 Total Incidents", len(incident_images))

# =========================
# ENVIRONMENT CHECK
# =========================

is_cloud = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true"

# =========================
# CLOUD MODE
# =========================

if is_cloud:

    st.warning("""
⚠ Webcam is disabled on Streamlit Cloud.

Why?
- Cloud servers cannot access your laptop webcam.

Use this app locally for:
✅ Live webcam detection

Use cloud deployment for:
✅ Portfolio showcase
✅ Dashboard preview
✅ Incident viewing
""")

# =========================
# LOCAL WEBCAM MODE
# =========================

else:

    frame_placeholder = st.empty()

    if start_monitoring:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot access webcam")
            st.stop()

        while True:

            ret, frame = cap.read()

            if not ret:
                st.error("❌ Failed to read webcam")
                break

            # YOLO Prediction
            results = model(frame)

            detected = False
            detected_label = ""

            for box in results[0].boxes:

                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[cls_id]

                if confidence > 0.60:

                    if class_name.lower() in ["fire", "smoke"]:

                        detected = True
                        detected_label = class_name

                        break

            # Draw results
            annotated_frame = results[0].plot()

            frame_rgb = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

            # Status
            if detected:

                status_box.error(
                    f"🚨 {detected_label.upper()} DETECTED"
                )

                # Save incident
                timestamp = time.strftime("%Y%m%d-%H%M%S")

                image_path = os.path.join(
                    SAVE_DIR,
                    f"{detected_label}_{timestamp}.jpg"
                )

                cv2.imwrite(image_path, frame)

            else:

                status_box.success(
                    "✅ SAFE - NO FIRE / SMOKE"
                )

            # Stop Button
            if stop_monitoring:
                break

        cap.release()

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