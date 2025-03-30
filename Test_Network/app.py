import cv2
import av
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import model_testc

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.initialized = False

    def _initialize(self):
        # Create a new graph and load the model once
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Define the placeholder for one image (batch_size=1, 96x96, 3 channels)
            self.x = tf.placeholder(tf.float32, shape=[1, 96, 96, 3])
            # Build the model (set dropout keep probability to 1.0 for inference)
            self.logits = model_testc.inference(self.x, batch_size=1, n_classes=2, dropout=1.0)
            self.logits = tf.nn.softmax(self.logits)
            self.saver = tf.train.Saver()
        # Create a session for the loaded graph
        self.sess = tf.Session(graph=self.graph)
        # Restore the trained model from the checkpoint directory
        logs_train_dir = 'D:\\train_myself\\model_testc\\'
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(f"Model restored from: {ckpt.model_checkpoint_path}")
        else:
            print("No checkpoint file found.")
        self.initialized = True

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        # Initialize the model once
        if not self.initialized:
            self._initialize()

        # Convert the incoming frame to a numpy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # If at least one face is detected, take the first face
            (x, y, w, h) = faces[0]
            # Draw a rectangle around the face for visualization (optional)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Crop the face region from the frame
            face_img = img[y:y+h, x:x+w]
            # Resize the cropped face to 96x96 for the model
            face_img = cv2.resize(face_img, (96, 96))
            
            # Preprocess the face image:
            # Convert BGR to RGB
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            # Convert to PIL image for consistency with training pipeline
            pil_img = Image.fromarray(rgb_face)
            # Convert back to numpy array and cast to float32
            input_img = np.array(pil_img).astype(np.float32)
            # Per-image standardization
            mean = np.mean(input_img)
            std = np.std(input_img)
            input_img = (input_img - mean) / (std + 1e-6)
            # Expand dimensions to create a batch of 1
            input_img_exp = np.expand_dims(input_img, axis=0)
            
            # Run inference on the preprocessed face image
            pred = self.sess.run(self.logits, feed_dict={self.x: input_img_exp})
            label = np.argmax(pred, axis=1)[0]
            conf = pred[0][label]
            # Determine label text based on prediction (assuming 1 = Real, 0 = Fake)
            label_text = "Real" if label == 1 else "Fake"
            # Choose text color: Green for "Real", Red for "Fake"
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            # Overlay the prediction on the original frame near the detected face
            cv2.putText(img, f"{label_text}: {conf:.2f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            # If no face is detected, overlay a warning message
            cv2.putText(img, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img

# Start the Streamlit app with the WebRTC streamer
webrtc_streamer(key="example", video_transformer_factory=FaceDetectionTransformer)
