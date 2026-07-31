import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model.h5")

# Labels
labels = ["A", "B", "C"]

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

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

            prediction = model.predict(data)
            class_id = np.argmax(prediction)
            label = labels[class_id]

            # Show prediction on screen
            cv2.putText(img, f"Prediction: {label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()