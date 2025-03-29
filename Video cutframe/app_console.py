from cropper import Cropper
import cv2

def face_crop(input_image,target_width,target_height):
    
    cropper = Cropper()
    
    target_image = cropper.crop(input_image, target_width, target_height)
    if target_image is None:
        print ('Cropping failed.')

    return target_image