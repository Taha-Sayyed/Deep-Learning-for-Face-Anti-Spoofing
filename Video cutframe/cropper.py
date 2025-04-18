import sys
from detector import Detector

# ----------------------------------------------------------------------------

class Cropper(object):
    """Cropper"""
    def __init__(self):
        super(Cropper, self).__init__()
        self.detector = Detector()

    @staticmethod
    def _bounding_rect(faces):
        top,    left  =  sys.maxsize,  sys.maxsize
        bottom, right = -sys.maxsize, -sys.maxsize
        for (x, y, w, h) in faces:
            if x < left:
                left = x
            if x+w > right:
                right = x+w
            if y < top:
                top = y
            if y+h > bottom:
                bottom = y+h
        return top, left, bottom, right

    def crop(self, img, target_width, target_height):
        original_height, original_width = img.shape[:2]
        faces = self.detector.detect_faces(img)
        
        if len(faces) == 0:  # no detected faces
            # If no face detected, use the center of the image
            target_center_x = original_width // 2
            target_center_y = original_height // 2
        else:
            # Face detected, get bounding box
            top, left, bottom, right = self._bounding_rect(faces)
            
            # Calculate the current face size
            face_width = right - left
            face_height = bottom - top
            
            # Calculate center of the face
            target_center_x = (left + right) // 2
            target_center_y = (top + bottom) // 2
            
            # Calculate a larger area to ensure the entire face is captured
            # Add more margin (1.5x the size of the face area)
            target_width = max(target_width, int(face_width * 1.5))
            target_height = max(target_height, int(face_height * 1.5))
        
        # Calculate the crop boundaries
        target_left = max(0, target_center_x - target_width // 2)
        target_right = min(original_width, target_left + target_width)
        target_top = max(0, target_center_y - target_height // 2)
        target_bottom = min(original_height, target_top + target_height)
        
        # Adjust if we're at the image boundaries
        if target_right >= original_width:
            target_left = max(0, original_width - target_width)
            target_right = original_width
        
        if target_bottom >= original_height:
            target_top = max(0, original_height - target_height)
            target_bottom = original_height
            
        # Return the cropped image
        return img[target_top:target_bottom, target_left:target_right]

# ----------------------------------------------------------------------------