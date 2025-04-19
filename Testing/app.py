import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import time

# Import our model utility functions
from model_utils import FaceAntiSpoofModel, detect_face

# Set page config
st.set_page_config(
    page_title="Face Anti-Spoofing Detection",
    page_icon="🔒",
    layout="centered"
)

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = False

def apply_label_to_frame(frame, label, confidence):
    """Apply prediction label to the video frame"""
    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    
    # Set color based on prediction
    if label == "REAL":
        color = (0, 255, 0)  # Green for real
    else:
        color = (0, 0, 255)  # Red for fake
    
    # Create the text to display
    text = f"{label}: {confidence:.2f}"
    
    # Calculate text position
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = 10
    text_y = 30
    
    # Draw background rectangle
    cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
    return frame

def start_webcam():
    st.session_state.running = True
    
def stop_webcam():
    st.session_state.running = False

def main():
    # Set up page title and description
    st.title("Face Anti-Spoofing Detection")
    st.markdown("This application detects whether a face is real or a spoofing attack.")
    
    # Model selection dropdown for multiple models
    model_options = {
        "Best Model (Accuracy: 100%)": "../model_checkpoints/model.08-1.00.keras",
        "Model (Accuracy: 93%)": "../model_checkpoints/model.07-0.93.keras",
        "Final Model": "../model_checkpoints/final_model.keras"
    }
    selected_model = st.selectbox("Select Model", list(model_options.keys()))
    model_path = model_options[selected_model]
    
    # Load the model
    with st.spinner("Loading model..."):
        face_model = FaceAntiSpoofModel(model_path)
        
        if face_model.model is None:
            st.error("Failed to load model. Please check model path and try again.")
            return
    
    st.success("Model loaded successfully!")
    
    # Create start/stop buttons
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button("START", on_click=start_webcam, disabled=st.session_state.running)
    
    with col2:
        stop_button = st.button("STOP", on_click=stop_webcam, disabled=not st.session_state.running)
    
    # Placeholder for webcam feed
    stframe = st.empty()
    
    # Process status placeholder
    status_placeholder = st.empty()
    
    # Initialize webcam capture
    cap = None
    
    if st.session_state.running:
        status_placeholder.info("Starting webcam...")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            status_placeholder.error("Could not open webcam. Please check your camera connection.")
            stop_webcam()
            st.experimental_rerun()
        
        status_placeholder.success("Webcam started. Detecting faces...")
        
    # Main webcam loop
    while st.session_state.running:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            status_placeholder.error("Could not read frame from webcam.")
            stop_webcam()
            break
        
        # Detect face in the frame
        frame_with_rect, face_roi = detect_face(frame)
        
        # If face detected, predict whether it's real or fake
        if face_roi is not None:
            # Predict
            label, confidence = face_model.predict(face_roi)
            
            # Apply label to frame
            frame_with_label = apply_label_to_frame(frame_with_rect, label, confidence)
            
            # Convert from BGR to RGB for display
            rgb_frame = cv2.cvtColor(frame_with_label, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            stframe.image(rgb_frame, channels="RGB", use_container_width=True)
        else:
            # If no face detected, just display the frame with a message
            text = "No face detected"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Convert from BGR to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            stframe.image(rgb_frame, channels="RGB", use_container_width=True)
        
        # Small pause to reduce CPU usage
        time.sleep(0.01)
        
    # Release webcam if it was initialized
    if cap is not None:
        cap.release()
        status_placeholder.info("Webcam stopped.")

if __name__ == "__main__":
    main() 