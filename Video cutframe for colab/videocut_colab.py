#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face extraction for anti-spoofing - Colab version
"""

import cv2
import os
import app_console
import time
from google.colab import files
from google.colab import drive
import shutil
import glob

def extract_frames(video_dir, store_dir, max_videos=240):
    """
    Extract face frames from videos and save as PNG files
    
    Args:
        video_dir: Directory containing the video files (numbered 1.avi, 2.avi, etc.)
        store_dir: Directory to store the extracted frames
        max_videos: Maximum number of videos to process (default: 240)
    """
    # Ensure output directory exists
    os.makedirs(store_dir, exist_ok=True)

    real_count = 0
    fake_count = 0
    skipped_videos = []
    extracted_videos = []

    print(f"Starting frame extraction for {max_videos} videos from {video_dir}")
    print(f"Storing frames in {store_dir}")

    # Process all videos
    for i in range(1, max_videos + 1):
        start_time = time.time()
        video_path = os.path.join(video_dir, f"{i}.avi")
        
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
                                output_path = os.path.join(store_dir, f"real.{real_count}.png")
                                cv2.imwrite(output_path, target_image)
                                real_count += 1
                            else:
                                output_path = os.path.join(store_dir, f"fake.{fake_count}.png")
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
    
    return {
        'real_frames': real_count,
        'fake_frames': fake_count,
        'total_frames': real_count + fake_count,
        'videos_processed': len(extracted_videos),
        'videos_skipped': len(skipped_videos)
    }

def mount_drive():
    """Mount Google Drive to access files"""
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
    print("Google Drive mounted.")

def upload_videos():
    """Upload videos from local machine to Colab"""
    print("Please upload your video files (numbered as 1.avi, 2.avi, etc.)")
    uploaded = files.upload()
    return list(uploaded.keys())

def download_results(store_dir):
    """Download extracted frames as a zip file"""
    if not os.path.exists(store_dir):
        print(f"Error: {store_dir} does not exist")
        return
        
    # Count files
    frame_files = glob.glob(os.path.join(store_dir, "*.png"))
    if len(frame_files) == 0:
        print("No frames found to download")
        return
        
    # Create zip file
    zip_path = "/content/extracted_frames.zip"
    shutil.make_archive("/content/extracted_frames", 'zip', store_dir)
    
    # Download the zip file
    print(f"Downloading {len(frame_files)} frames as a zip file...")
    files.download(zip_path)

# Example usage in a notebook:
"""
# Run this in your Colab notebook

# 1. Import the module
from videocut_colab import extract_frames, mount_drive, download_results

# 2. Mount Google Drive (if videos are stored there)
mount_drive()

# 3. Set paths
VIDEO_DIR = "/content/drive/MyDrive/your_videos_folder"  # Change to your video directory
STORE_DIR = "/content/extracted_frames"                 # Where to save the frames

# 4. Extract frames
results = extract_frames(VIDEO_DIR, STORE_DIR)

# 5. Download the extracted frames
download_results(STORE_DIR)
""" 