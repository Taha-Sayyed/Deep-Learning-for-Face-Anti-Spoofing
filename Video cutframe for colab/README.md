# Face Extraction for Anti-Spoofing in Google Colab

This directory contains Python code for extracting face frames from videos for anti-spoofing, adapted for use in Google Colab.

## Files Description

- `app_console.py`: Interface for face cropping functionality
- `cropper.py`: Handles the cropping of faces from video frames
- `detector.py`: Handles face detection using OpenCV's Haar Cascade
- `videocut_colab.py`: Main script for extracting faces from videos in Google Colab
- `Face_Extraction_Colab.ipynb`: Sample Jupyter notebook for Google Colab

## How to Use in Google Colab

1. Upload all the Python files (`app_console.py`, `cropper.py`, `detector.py`, and `videocut_colab.py`) to your Google Colab session.

2. Upload the `Face_Extraction_Colab.ipynb` notebook to Google Colab or create a new notebook and follow similar steps.

3. If your videos are in Google Drive:
   - Mount your Google Drive
   - Set `VIDEO_DIR` to point to your videos folder in Google Drive

4. If your videos are on your local machine:
   - Upload them directly to Colab using the `files.upload()` function

5. Run the extraction process, which will:
   - Process videos named as `1.avi`, `2.avi`, etc.
   - Label frames from videos 1-61 as "real"
   - Label frames from videos 62-240 as "fake"
   - Extract approximately 20 frames from each video

6. Download the extracted frames as a zip file.

## Requirements

The code requires the following Python packages:
- OpenCV (cv2)
- NumPy
- Google Colab integration packages (automatically available in Colab)

## Key Improvements for Colab

1. **No Hard-Coded Paths**: All paths are configurable and use `os.path.join()` for better cross-platform compatibility.

2. **Automated Cascade File Download**: The `detector.py` file automatically downloads the Haar Cascade file if it's not found.

3. **Google Drive Integration**: Option to access videos from Google Drive.

4. **Upload/Download Features**: Support for uploading videos and downloading extracted frames.

5. **Flexible Configuration**: Easy-to-configure parameters for different datasets.

## Video Naming and Labeling

- Videos should be named as `1.avi`, `2.avi`, ..., `240.avi`
- Videos 1-61 will be labeled as "real"
- Videos 62-240 will be labeled as "fake"

## Output Format

Extracted frames will be named as:
- `real.0.png`, `real.1.png`, ... for real faces
- `fake.0.png`, `fake.1.png`, ... for fake faces 