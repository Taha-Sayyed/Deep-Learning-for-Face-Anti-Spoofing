# Face Anti-Spoofing Detection App

This application uses deep learning to detect whether a face is real or a spoofing attack (e.g., photo, video, or mask) in real-time using a webcam.

## Requirements

- Python 3.7+
- TensorFlow 2.4+
- OpenCV
- Streamlit

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. The application will open in your default web browser.

3. Select a model from the dropdown menu.

4. Click the "START" button to begin face detection and anti-spoofing analysis.

5. Position your face in front of the webcam. The application will:
   - Detect your face
   - Analyze whether it's real or fake
   - Display the result on the screen

6. Click "STOP" to end the webcam session.

## Model Architecture

The model uses a CNN architecture with the following structure:

- Input: 96 × 96 × 3 RGB image
- Convolutional layers with max-pooling and dropout layers
- Multiple fully connected layers
- Output: Binary classification (Real/Fake)

## Notes

- For best results, ensure good lighting conditions
- Face should be clearly visible and not obscured
- Model performance may vary depending on the selected checkpoint 