import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Start camera
cap = cv2.VideoCapture(0)

# Change label for A, B, C
label = "Y"

# Create folder if not exists
if not os.path.exists(f"dataset/{label}"):
    os.makedirs(f"dataset/{label}")

count = 0
frame_count = 0   # to control saving speed

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            data = []

            for lm in handLms.landmark:
                data.append(lm.x)
                data.append(lm.y)

            frame_count += 1

            # 🔥 AUTO SAVE every 10 frames
            if frame_count % 10 == 0:
                np.save(f"dataset/{label}/{count}.npy", data)
                count += 1
                print("Saved:", count)

    # Show instructions
    cv2.putText(img, f"Collecting {label} | Saved: {count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Collect Data", img)

    # Press ESC to stop
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()