import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3

# Load trained model and label classes
model = tf.keras.models.load_model("sign_language_model.h5")
label_classes = np.load("label_classes.npy", allow_pickle=True)  # Ensure allow_pickle=True

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize speech engine
engine = pyttsx3.init()

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process image with Mediapipe Hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])  # Use x, y, z coordinates

            # Normalize landmarks (if required)
            landmark_array = np.array(landmark_list).reshape(1, -1)
            landmark_array = (landmark_array - np.min(landmark_array)) / (
                        np.max(landmark_array) - np.min(landmark_array))

            # Make prediction
            prediction = model.predict(landmark_array)
            class_index = np.argmax(prediction)
            predicted_label = label_classes[class_index]

            # Speak prediction
            engine.say(predicted_label)
            engine.runAndWait()

            # Display prediction on screen
            cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Sign Language Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
