#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Extraction for Anti-Spoofing in Google Colab

This script demonstrates how to extract face frames from video files for anti-spoofing using Google Colab.
You can run this in Colab by uploading all the required files and executing the cells.
"""

# SECTION 1: Setup Environment
# ===========================

# Import necessary libraries
import os
import glob
import cv2
import time
from google.colab import files, drive
import matplotlib.pyplot as plt
import random

print("OpenCV version:", cv2.__version__)
print("Setup complete!")

# SECTION 2: Upload Code Files
# ===========================
# You need to upload these files to Colab:
# 1. app_console.py
# 2. cropper.py
# 3. detector.py
# 4. videocut_colab.py

print("Please upload the required Python files (app_console.py, cropper.py, detector.py, videocut_colab.py)")
uploaded = files.upload()
print(f"\nUploaded {len(uploaded)} files: {list(uploaded.keys())}")

# SECTION 3: Mount Google Drive (Optional)
# ======================================
# Import the face extraction module
from videocut_colab import extract_frames, mount_drive, upload_videos, download_results

# Mount Google Drive
print("Do you want to mount Google Drive? (yes/no)")
if input().lower() == "yes":
    mount_drive()

# SECTION 4: Define Video and Output Directories
# ===========================================
# Option 1: If your videos are in Google Drive
# VIDEO_DIR = "/content/drive/MyDrive/your_videos_folder"  # Change to your video directory

# Option 2: If you want to upload videos to Colab's temporary storage
VIDEO_DIR = "/content/videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# Directory to store extracted frames
STORE_DIR = "/content/extracted_frames"
os.makedirs(STORE_DIR, exist_ok=True)

print(f"Video directory: {VIDEO_DIR}")
print(f"Frame storage directory: {STORE_DIR}")

# SECTION 5: Upload Videos (Option 2)
# =================================
# Skip this section if using Google Drive

print("Do you want to upload videos? (yes/no)")
if input().lower() == "yes":
    print("Please upload your video files (should be named 1.avi, 2.avi, etc.)")
    uploaded_videos = files.upload()
    
    # Save uploaded videos to the videos directory
    for filename, content in uploaded_videos.items():
        with open(os.path.join(VIDEO_DIR, filename), 'wb') as f:
            f.write(content)
            
    print(f"\nUploaded {len(uploaded_videos)} videos to {VIDEO_DIR}")
    print(f"Video files: {list(uploaded_videos.keys())}")

# SECTION 6: Extract Faces from Videos
# ==================================
# This will process all videos named 1.avi, 2.avi, etc. up to 240.avi.
# - Videos 1-61 will be labeled as "real"
# - Videos 62-240 will be labeled as "fake"

print("Do you want to start face extraction? (yes/no)")
if input().lower() == "yes":
    # You can adjust max_videos parameter if you have fewer videos
    max_videos = int(input("Enter maximum number of videos to process (default is 240): ") or "240")
    
    # Extract faces from videos
    results = extract_frames(VIDEO_DIR, STORE_DIR, max_videos=max_videos)
    
    # Show results
    print(f"\nExtraction completed with {results['total_frames']} frames extracted:")
    print(f"  - Real frames: {results['real_frames']}")
    print(f"  - Fake frames: {results['fake_frames']}")
    print(f"  - Videos processed: {results['videos_processed']}")
    print(f"  - Videos skipped: {results['videos_skipped']}")

# SECTION 7: Download Extracted Frames
# ==================================
print("Do you want to download the extracted frames? (yes/no)")
if input().lower() == "yes":
    # Download the extracted frames as a zip file
    download_results(STORE_DIR)

# SECTION 8: View Some Sample Frames (Optional)
# ==========================================
print("Do you want to view some sample frames? (yes/no)")
if input().lower() == "yes":
    # Find all PNG files in the output directory
    frame_files = glob.glob(os.path.join(STORE_DIR, "*.png"))
    
    # Display some random samples if files exist
    if frame_files:
        # Select up to 5 random samples
        samples = random.sample(frame_files, min(5, len(frame_files)))
        
        plt.figure(figsize=(15, 4))
        for i, sample in enumerate(samples):
            plt.subplot(1, len(samples), i+1)
            img = cv2.imread(sample)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color display
            plt.imshow(img_rgb)
            plt.title(os.path.basename(sample))
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("No frames found to display")

print("\n--- Script completed ---") 