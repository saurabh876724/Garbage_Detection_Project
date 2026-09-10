# ==========================================================
# FINAL YEAR PROJECT
# SMART GARBAGE DETECTION USING YOLOv8
# ==========================================================

import cv2
import os
from ultralytics import YOLO


# ==========================================================
# STEP 1 : CHECK MODEL PATH
# ==========================================================

MODEL_PATH = "runs/detect/train6/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: Trained model not found.")
    print("Please check the path:", MODEL_PATH)
    print("Go to runs/detect/ folder and find correct train folder.")
    exit()


# ==========================================================
# STEP 2 : LOAD YOLO MODEL
# ==========================================================

print("Loading YOLO Garbage Detection Model...")

try:
    model = YOLO(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()


# ==========================================================
# STEP 3 : DEFINE GARBAGE CLASSES
# ==========================================================

garbage_classes = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]


# ==========================================================
# STEP 4 : START WEBCAM
# ==========================================================

print("Starting Webcam...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot access webcam")
    exit()

print("🚀 Garbage Detection Started")
print("Press 'Q' to exit")


# ==========================================================
# STEP 5 : REAL TIME DETECTION LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Frame capture failed")
        break

    try:

        # Run YOLO detection
        results = model(frame, conf=0.5)

        garbage_detected = False

        for r in results:

            boxes = r.boxes

            if boxes is None:
                continue

            for box in boxes:

                cls = int(box.cls[0])
                label = model.names[cls]
                confidence = float(box.conf[0])

                if label in garbage_classes:

                    garbage_detected = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    text = f"{label} ({confidence:.2f})"

                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

        if garbage_detected:
            cv2.putText(
                frame,
                "GARBAGE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

    except Exception as e:
        print("Detection Error:", e)

    cv2.imshow("Smart Garbage Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Closing Application...")
        break


# ==========================================================
# STEP 6 : CLEANUP
# ==========================================================

cap.release()
cv2.destroyAllWindows()

print("✅ Program Ended Successfully")