import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Labels A–Z
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Start camera
cap = cv2.VideoCapture(0)

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

            data = np.array(data).reshape(1, -1)

            # Prediction
            pred = model.predict(data)[0]
            label = labels[pred]

            # 🔥 Confidence calculation
            proba = model.predict_proba(data)
            confidence = max(proba[0])

            # 🔥 Show prediction + confidence
            if confidence > 0.7:   # optional filter
                cv2.putText(img, f"{label} ({confidence:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", img)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()