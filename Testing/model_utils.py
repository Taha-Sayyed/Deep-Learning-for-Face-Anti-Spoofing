import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

class FaceAntiSpoofModel:
    """Class to handle model operations for face anti-spoofing detection"""
    
    def __init__(self, model_path="../model_checkpoints/model.08-1.00.keras"):
        """Initialize the model
        
        Args:
            model_path (str): Path to the saved model
        """
        self.img_width = 96
        self.img_height = 96
        self.n_classes = 2
        self.model_path = model_path
        self.model = None
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the trained model from checkpoint"""
        if not os.path.exists(self.model_path):
            print(f"Error: Model not found at {self.model_path}")
            return False
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tensor: Preprocessed image tensor
        """
        # Resize to required dimensions
        image = cv2.resize(image, (self.img_width, self.img_height))
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # If RGBA, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Standardize the image
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, self.img_width, self.img_height, 3])
        
        return image
    
    def predict(self, image):
        """Predict whether the face is real or fake
        
        Args:
            image (numpy.ndarray): Input face image
            
        Returns:
            tuple: (result, confidence) where result is "REAL" or "FAKE"
        """
        if self.model is None:
            print("Error: Model not loaded")
            return "UNKNOWN", 0.0
        
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        prediction = self.model.predict(processed_image, verbose=0)
        
        # Get the predicted class
        max_index = np.argmax(prediction[0])
        confidence = prediction[0][max_index]
        
        # 0: Fake, 1: Real (based on the training labels)
        result = "REAL" if max_index == 1 else "FAKE"
        
        return result, confidence

def detect_face(frame):
    """Detect face in the frame using OpenCV's Haar Cascade
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        tuple: (frame_with_rect, face_roi) where face_roi is the detected face region
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # If no face detected, return the original frame
    if len(faces) == 0:
        return frame, None
    
    # Get the largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Draw rectangle around the face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Extract the face ROI
    face_roi = frame[y:y+h, x:x+w]
    
    return frame, face_roi 