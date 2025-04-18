import cv2
import os

# ----------------------------------------------------------------------------

DEFAULT_DATA_FILENAME = os.path.join(
    os.path.split(os.path.realpath(__file__))[0],
    'haarcascade_frontalface_default.xml'
    )

class Detector(object):
    """Detector"""
    def __init__(self, data_filename = DEFAULT_DATA_FILENAME):
        super(Detector, self).__init__()
        self.face_cascade = cv2.CascadeClassifier(data_filename)

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