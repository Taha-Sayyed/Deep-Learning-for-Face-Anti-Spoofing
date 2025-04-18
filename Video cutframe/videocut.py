import cv2
import os
import app_console

Video_dir = r'C:\Users\Hootone Tahir\Desktop\TE Project\Deep-Learning-for-Face-Anti-Spoofing\Revised_train_release\\'
store_dir = r'C:\Users\Hootone Tahir\Desktop\TE Project\Deep-Learning-for-Face-Anti-Spoofing\frames\train_set\\'

# Ensure output directory exists
os.makedirs(store_dir, exist_ok=True)

real_count = 0
fake_count = 0

for i in range(1, 240):
    video_path = Video_dir + str(i) + '.avi'
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist. Skipping.")
        continue
    
    # First pass: count frames
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        print(f"Could not open video {video_path}. Skipping.")
        continue
        
    frame_count = 0
    while True:
        ret, frame = vc.read()
        if not ret or frame is None:
            break
        frame_count += 1
    vc.release()
    
    print(f"Video {i}.avi has {frame_count} frames")
    
    # Skip videos with no frames
    if frame_count == 0:
        print(f"Video {i}.avi has no frames. Skipping.")
        continue
    
    # Calculate frame sampling interval
    gap = max(frame_count // 20, 1)  # Ensure gap is at least 1
    
    # Second pass: extract frames
    vc = cv2.VideoCapture(video_path)
    frame_index = 0
    
    while True:
        ret, frame = vc.read()
        if not ret or frame is None:
            break
            
        if frame_index % gap == 0:
            try:
                # Use 400x400 dimensions to ensure more of the face is captured
                # The cropper will handle scaling based on actual face size
                target_image = app_console.face_crop(frame, 400, 400)
                
                if target_image is not None:
                    # Make sure the image is not too small
                    if target_image.shape[0] > 100 and target_image.shape[1] > 100:
                        if i <= 61:
                            output_path = store_dir + 'real.' + str(real_count) + '.png'
                            cv2.imwrite(output_path, target_image)
                            real_count += 1
                        else:
                            output_path = store_dir + 'fake.' + str(fake_count) + '.png'
                            cv2.imwrite(output_path, target_image)
                            fake_count += 1
                    else:
                        print(f"Cropped image too small for frame {frame_index} in video {i}.avi")
                else:
                    print(f"Face cropping failed for frame {frame_index} in video {i}.avi")
            except Exception as e:
                print(f"Error processing frame {frame_index} in video {i}.avi: {str(e)}")
                
        frame_index += 1
        
    vc.release()
    print(f"Processed video {i}.avi: extracted {frame_index//gap} frames")

print(f"Total frames extracted: {real_count} real, {fake_count} fake")
