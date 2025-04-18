import cv2
import os
import app_console
import time

# Define paths with proper escape for Windows
Video_dir = r'C:\Users\Hootone Tahir\Desktop\TE Project\Deep-Learning-for-Face-Anti-Spoofing\Revised_train_release\\'
store_dir = r'C:\Users\Hootone Tahir\Desktop\TE Project\Deep-Learning-for-Face-Anti-Spoofing\frames\train_set\\'

# Ensure output directory exists
os.makedirs(store_dir, exist_ok=True)

real_count = 0
fake_count = 0
skipped_videos = []
extracted_videos = []

print(f"Starting frame extraction for 240 videos from {Video_dir}")
print(f"Storing frames in {store_dir}")

# Process all 240 videos
for i in range(1, 241):  # Changed to 241 to include video 240
    start_time = time.time()
    video_path = Video_dir + str(i) + '.avi'
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist. Skipping.")
        skipped_videos.append(i)
        continue
    
    print(f"Processing video {i}.avi...")
    
    # First pass: count frames
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        print(f"Could not open video {video_path}. Skipping.")
        skipped_videos.append(i)
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
        skipped_videos.append(i)
        continue
    
    # Calculate frame sampling interval - extract 20 frames from each video
    gap = max(frame_count // 20, 1)  # Ensure gap is at least 1
    
    # Second pass: extract frames
    vc = cv2.VideoCapture(video_path)
    frames_extracted = 0
    frame_index = 0
    
    while True:
        ret, frame = vc.read()
        if not ret or frame is None:
            break
            
        # Extract every gap-th frame
        if frame_index % gap == 0:
            try:
                # Use 400x400 dimensions to ensure more of the face is captured
                target_image = app_console.face_crop(frame, 400, 400)
                
                if target_image is not None:
                    # Make sure the image is not too small
                    if target_image.shape[0] > 100 and target_image.shape[1] > 100:
                        # Videos 1-61 are labeled as real, 62-240 as fake
                        if i <= 61:
                            output_path = store_dir + 'real.' + str(real_count) + '.png'
                            cv2.imwrite(output_path, target_image)
                            real_count += 1
                        else:
                            output_path = store_dir + 'fake.' + str(fake_count) + '.png'
                            cv2.imwrite(output_path, target_image)
                            fake_count += 1
                        
                        frames_extracted += 1
                    else:
                        print(f"Cropped image too small for frame {frame_index} in video {i}.avi")
                else:
                    print(f"Face cropping failed for frame {frame_index} in video {i}.avi")
            except Exception as e:
                print(f"Error processing frame {frame_index} in video {i}.avi: {str(e)}")
                
        frame_index += 1
        
    vc.release()
    
    elapsed_time = time.time() - start_time
    
    if frames_extracted > 0:
        extracted_videos.append(i)
        print(f"Processed video {i}.avi: extracted {frames_extracted} frames in {elapsed_time:.2f} seconds")
    else:
        skipped_videos.append(i)
        print(f"No frames were extracted from video {i}.avi")

# Final summary
print("\n----- Extraction Summary -----")
print(f"Total frames extracted: {real_count + fake_count}")
print(f"Real frames: {real_count}")
print(f"Fake frames: {fake_count}")
print(f"Videos processed successfully: {len(extracted_videos)}")
print(f"Videos skipped: {len(skipped_videos)}")

if len(skipped_videos) > 0:
    print(f"Skipped video numbers: {skipped_videos}")

print("\nFrame extraction completed!")
