import cv2
import os
import urllib.request

# ----------------------------------------------------------------------------

# For Google Colab: Download the cascade file if needed
def ensure_cascade_file_exists(file_path="/content/haarcascade_frontalface_default.xml"):
    """Ensure the cascade file exists in Colab, downloading it if necessary"""
    if not os.path.exists(file_path):
        print(f"Downloading cascade file to {file_path}...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, file_path)
        print("Download complete!")
    return file_path

# Default location for Colab
DEFAULT_DATA_FILENAME = ensure_cascade_file_exists()

class Detector(object):
    """Detector"""
    def __init__(self, data_filename=DEFAULT_DATA_FILENAME):
        super(Detector, self).__init__()
        self.face_cascade = cv2.CascadeClassifier(data_filename)
        
        # Verify the file was loaded correctly
        if self.face_cascade.empty():
            raise ValueError(f"Error: Could not load cascade classifier from {data_filename}")
            
    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Improved parameters for better face detection
        # minNeighbors: Higher value = less detections but higher quality
        # scaleFactor: Closer to 1 = better detection but slower
        # minSize: Minimum size of face to detect
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces

# ---------------------------------------------------------------------------- 