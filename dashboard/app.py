import streamlit as st
import cv2
import os
import time
import sys
from datetime import datetime
from ultralytics import YOLO

# Fix imports
sys.path.append(os.path.abspath("."))

from alerts.email_alert import send_email_alert
from alerts.telegram_alert import send_telegram_alert

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Industrial Fire Detection System",
    page_icon="🔥",
    layout="wide"
)

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\shash\runs\detect\fire_smoke_detector-2\weights\best.pt"

SAVE_DIR = "incidents"
ALERT_COOLDOWN = 60

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)

# ---------------- UI ----------------
st.title("🔥 Industrial Fire & Smoke Detection Dashboard")
st.markdown("### Professional Real-Time AI Monitoring System")

# Sidebar
st.sidebar.title("Monitoring Controls")

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if st.sidebar.button("▶ Start Monitoring"):
    st.session_state.monitoring = True

if st.sidebar.button("⏹ Stop Monitoring"):
    st.session_state.monitoring = False

# Dashboard Metrics
incident_count = len(os.listdir(SAVE_DIR))

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Incidents", incident_count)

with col2:
    st.metric("System Status", "ACTIVE")

with col3:
    st.metric("Alert System", "Email + Telegram")

# Live placeholders
frame_placeholder = st.empty()
status_placeholder = st.empty()
confidence_placeholder = st.empty()

# ---------------- LIVE MONITORING ----------------
if st.session_state.monitoring:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Camera failed to open")
        st.stop()

    last_alert_time = 0

    while st.session_state.monitoring:

        ret, frame = cap.read()

        if not ret:
            st.error("❌ Failed to capture frame")
            break

        # ---------------- DETECTION ----------------
        results = model(
            frame,
            conf=0.60,
            iou=0.45,
            imgsz=640
        )

        detected = False
        detected_label = ""
        detected_confidence = 0

        frame_h, frame_w = frame.shape[:2]

        for box in results[0].boxes:

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height

            # Ignore tiny false detections
            if box_area < 15000:
                continue

            # Ignore webcam center face region
            center_x = frame_w // 2
            center_y = frame_h // 2

            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2

            if (
                center_x - 150 < box_center_x < center_x + 150
                and center_y - 150 < box_center_y < center_y + 150
            ):
                continue

            # Final detection check
            if (
                class_name.lower() in ["fire", "smoke"]
                and confidence > 0.60
            ):

                detected = True
                detected_label = class_name
                detected_confidence = round(confidence * 100, 2)

                break

        # ---------------- DRAW RESULTS ----------------
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

        # ---------------- STATUS DISPLAY ----------------
        if detected:

            status_placeholder.error(
                f"🚨 {detected_label.upper()} DETECTED"
            )

            confidence_placeholder.warning(
                f"Confidence: {detected_confidence}%"
            )

        else:

            status_placeholder.success(
                "✅ SAFE - NO FIRE / SMOKE"
            )

            confidence_placeholder.info(
                "Monitoring Active"
            )

        # ---------------- ALERT LOGIC ----------------
        if detected and (time.time() - last_alert_time > ALERT_COOLDOWN):

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            image_path = os.path.join(
                SAVE_DIR,
                f"incident_{timestamp}.jpg"
            )

            # Save incident image
            cv2.imwrite(image_path, frame)

            # Email Alert
            send_email_alert(
                "🔥 FIRE / SMOKE ALERT",
                f"{detected_label} detected at {timestamp}"
            )

            # Telegram Alert
            send_telegram_alert(
                f"🔥 ALERT\n{detected_label} detected at {timestamp}"
            )

            st.warning(
                f"⚠ Alert Sent: {detected_label}"
            )

            last_alert_time = time.time()

    cap.release()

# ---------------- INCIDENT GALLERY ----------------
st.subheader("📸 Incident Gallery")

incident_images = os.listdir(SAVE_DIR)

if incident_images:

    cols = st.columns(3)

    for idx, img_name in enumerate(
        reversed(incident_images[-9:])
    ):

        img_path = os.path.join(
            SAVE_DIR,
            img_name
        )

        with cols[idx % 3]:
            st.image(
                img_path,
                caption=img_name
            )

else:
    st.info("No incidents recorded yet.")