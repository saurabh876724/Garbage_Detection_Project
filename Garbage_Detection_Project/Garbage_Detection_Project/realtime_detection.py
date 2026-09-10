# ==========================================================
# SMART GARBAGE DETECTION SYSTEM
# Final Year Project - YOLOv8 Real Time Detection
# ==========================================================

import cv2
import os
from ultralytics import YOLO


# ==========================================================
# STEP 1 : MODEL PATH
# ==========================================================

MODEL_PATH = "runs/detect/train6/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: Model not found")
    print("Check path:", MODEL_PATH)
    exit()

print("Loading YOLO Model...")

model = YOLO(MODEL_PATH)

print("Model Loaded Successfully")
print("Model Classes:", model.names)


# ==========================================================
# STEP 2 : START WEBCAM
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

print("🚀 Real Time Garbage Detection Started")
print("Press Q to exit")


# ==========================================================
# STEP 3 : REAL TIME DETECTION LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame capture error")
        break


    # YOLO Detection
    results = model(frame, conf=0.25)


    garbage_count = 0


    for r in results:

        boxes = r.boxes

        if boxes is None:
            continue


        for box in boxes:

            # class index
            cls = int(box.cls[0])

            # class label
            label = model.names[cls]

            # confidence
            confidence = float(box.conf[0])

            # bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])


            garbage_count += 1


            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )


            # Label
            text = f"{label} {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # ==========================================================
    # DISPLAY GARBAGE COUNT
    # ==========================================================

    cv2.putText(
        frame,
        f"Garbage Count: {garbage_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )


    # ==========================================================
    # AREA STATUS
    # ==========================================================

    if garbage_count > 0:

        cv2.putText(
            frame,
            "STATUS : DIRTY AREA",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            frame,
            "STATUS : CLEAN AREA",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )


    # ==========================================================
    # SHOW FRAME
    # ==========================================================

    cv2.imshow("Smart Garbage Detection System", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==========================================================
# STEP 4 : CLOSE PROGRAM
# ==========================================================

cap.release()
cv2.destroyAllWindows()

print("Program Closed Successfully")