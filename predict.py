import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3

# Load trained model and label classes
model = tf.keras.models.load_model("sign_language_model.h5")
label_classes = np.load("label_classes.npy")

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize speech engine
engine = pyttsx3.init()

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)
                landmark_list.append(lm.z)

            landmark_array = np.array(landmark_list).reshape(1, -1)
            prediction = model.predict(landmark_array)
            class_index = np.argmax(prediction)
            predicted_label = label_classes[class_index]

            # Speak prediction
            engine.say(predicted_label)
            engine.runAndWait()

            # Display prediction on screen
            cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
