import cv2
import os
import time
import sys
from datetime import datetime
from ultralytics import YOLO

# Fix import path
sys.path.append(os.path.abspath("."))

from alerts.email_alert import send_email_alert
from alerts.telegram_alert import send_telegram_alert

# Load trained YOLO model
model = YOLO(r"C:\Users\shash\runs\detect\fire_smoke_detector-2\weights\best.pt")

# Create incident folder
SAVE_DIR = "incidents"
os.makedirs(SAVE_DIR, exist_ok=True)

# Alert cooldown (seconds)
ALERT_COOLDOWN = 60
last_alert_time = 0

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera failed to open")
    exit()

print("✅ Industrial Fire Detection System Started...")
print("📷 Monitoring 24/7...")
print("Press Q to stop")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to capture frame")
        break

    # Run detection
    results = model(frame)

    fire_detected = False

    # Check detections
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name in ["Fire", "Smoke"]:
            fire_detected = True
            break

    # Alert logic
    if fire_detected and (time.time() - last_alert_time > ALERT_COOLDOWN):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        image_path = os.path.join(
            SAVE_DIR,
            f"incident_{timestamp}.jpg"
        )

        # Save incident image
        cv2.imwrite(image_path, frame)

        print(f"🔥 Incident saved: {image_path}")

        # Email alert
        send_email_alert(
            "🔥 FIRE/SMOKE ALERT",
            f"Fire or smoke detected at {timestamp}"
        )

        # Telegram alert
        send_telegram_alert(
            f"🔥 FIRE/SMOKE ALERT\nDetected at {timestamp}"
        )

        last_alert_time = time.time()

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Show window
    cv2.imshow("Industrial Fire Detection System", annotated_frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 System stopped")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()