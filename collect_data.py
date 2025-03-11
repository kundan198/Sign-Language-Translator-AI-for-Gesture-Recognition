import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Define dataset path
dataset_path = "asl_alphabet_train"
output_csv = "landmarks_data.csv"

# Prepare dataset list
data = []
labels = []

def extract_landmarks(image_path, label):
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Error loading image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)
                landmark_list.append(lm.z)

            data.append(landmark_list)
            labels.append(label)

            # Show image while processing
            cv2.putText(image, f"Processing: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Data Collection", image)
            cv2.waitKey(100)  # Pause for 100ms

# Process dataset with progress tracking
total_images = sum([len(files) for _, _, files in os.walk(dataset_path)])
processed_count = 0

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            extract_landmarks(image_path, label)
            processed_count += 1
            print(f"✅ Processed {processed_count}/{total_images} images")

# Convert to DataFrame and save
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv(output_csv, index=False)

# Cleanup
cv2.destroyAllWindows()

print(f"\n🎉 Data collection complete! Saved {len(data)} samples to {output_csv}")
