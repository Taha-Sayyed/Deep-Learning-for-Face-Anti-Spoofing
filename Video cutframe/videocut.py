import cv2
import os
import app_console

Video_dir = '/Users/taha/Desktop/Antispoofing Dataset/CASIA_faceAntisp/train_release/' 
store_dir = '/Users/taha/Desktop/Antispoofing Dataset/CASIA_faceAntisp/frame/train_release/'


real_count = 0
fake_count = 0

# List all subdirectories in Video_dir (each representing a person)
person_dirs = [d for d in os.listdir(Video_dir) if os.path.isdir(os.path.join(Video_dir, d))]


# Process each person directory
for person in person_dirs:
    person_path = os.path.join(Video_dir, person)
    # List all .avi files in this person's directory
    video_files = [f for f in os.listdir(person_path) if f.endswith('.avi')]
    
    for video_file in video_files:
        video_path = os.path.join(person_path, video_file)
        vc = cv2.VideoCapture(video_path)
        
        # Count total frames
        frame_count = 0
        flag, frame = vc.read()
        while flag:
            frame_count += 1
            flag, frame = vc.read()
        vc.release()
        
        if frame_count == 0:
            print(f"No frames found in {video_path}. Skipping...")
            continue
        
        print(f"{video_file} in folder {person} has {frame_count} frames.")
        gap = frame_count // 20  # Extract 20 frames evenly
        if gap == 0:
            gap = 1  # In case the video is very short
        
        c = 1  # Frame counter
        
        vc = cv2.VideoCapture(video_path)
        flag, frame = vc.read()
        while flag:
            if c % gap == 0:
                # Determine cropping dimensions based on person folder name and/or video file
                # Here, we assume that person folders "1" and "2" (and optionally "3") are 'real'
                # and folders "4", "5", etc., are 'fake'. Adjust these conditions as needed.
                if int(person) <= 3:
                    if int(person) <= 2:
                        target_image = app_console.face_crop(frame, 240, 240)
                    else:
                        target_image = app_console.face_crop(frame, 600, 800)
                    image_name = os.path.join(store_dir, f"real.{real_count}.png")
                    real_count += 1
                else:
                    if int(person) <= 9:
                        target_image = app_console.face_crop(frame, 240, 240)
                    else:
                        target_image = app_console.face_crop(frame, 600, 800)
                    image_name = os.path.join(store_dir, f"fake.{fake_count}.png")
                    fake_count += 1
                    
                cv2.imwrite(image_name, target_image)
            c += 1
            flag, frame = vc.read()
        vc.release()
        cv2.waitKey(1)