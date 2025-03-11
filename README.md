# Sign-Language-Translator-AI-for-Gesture-Recognition

This project enables **real-time ASL recognition** using deep learning.

This project enables real-time ASL (American Sign Language) recognition using deep learning and MediaPipe. The system captures hand gestures, extracts key features, and predicts the corresponding ASL alphabet.

Features

Uses MediaPipe for hand tracking.

Trains a CNN model for gesture classification.

Supports real-time predictions via webcam.

Outputs both text and speech for recognized gestures.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-repo/ASL-Recognition.git
   cd ASL-Recognition
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Running the Program

1. **Collect dataset (using MediaPipe to extract landmarks):**
   ```sh
   python collect_data.py
   ```

2. **Train the model:**
   ```sh
   python train_data.py
   ```

3. **Run the prediction (real-time or image-based):**
   ```sh
   python predict_data.py
   ```

## Notes

- Ensure you have a **webcam** connected for real-time predictions.
- If accuracy is low, consider **increasing dataset size** or **tweaking hyperparameters**.

If you have any questions or issues, feel free to ask! 🚀

