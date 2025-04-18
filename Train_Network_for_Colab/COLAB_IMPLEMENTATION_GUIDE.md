# Google Colab Implementation Guide for Face Anti-Spoofing Model Training

This guide will walk you through the process of implementing and running the face anti-spoofing model training script in Google Colab.

## Overview

The training script (`train_model_colab_script.py`) contains all the code needed to train a deep learning model for face anti-spoofing. Unlike the original approach with separate files, this consolidated script includes everything in a single file for easier implementation in Google Colab.

## Step-by-Step Instructions

### 1. Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook by clicking on `File → New notebook`

### 2. Upload the Training Script

1. In your first code cell, add the following code to upload the script:

```python
# Upload the training script
from google.colab import files
uploaded = files.upload()  # Upload train_model_colab_script.py
print("Uploaded:", list(uploaded.keys()))
```

2. Run this cell by clicking the play button or pressing Shift+Enter
3. Click the "Choose Files" button that appears
4. Select the `train_model_colab_script.py` file from your computer
5. Wait for the upload to complete

### 3. Enable GPU Acceleration (Recommended)

For faster training, enable GPU acceleration:

1. Click on `Runtime → Change runtime type`
2. Select `GPU` from the Hardware accelerator dropdown
3. Click `Save`

### 4. Run the Training Script

You have two options for running the script:

#### Option A: Run the entire script at once

Add this to a new code cell:

```python
%run train_model_colab_script.py
```

This will run the script interactively, asking you questions about mounting Google Drive, uploading data, etc.

#### Option B: Run the script section by section (Recommended)

This approach gives you more control over each step. Copy and paste each section from the original script to separate cells in your notebook.

1. **Setup Environment** (First Code Cell):

```python
# Copy SECTION 1 from train_model_colab_script.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import files, drive
import random
import zipfile
import glob

# Check TensorFlow version and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
```

2. **Configure Parameters** (Second Code Cell):

```python
# Copy SECTION 2 from train_model_colab_script.py
# Training parameters - Feel free to modify these
N_CLASSES = 2         # Number of classes (real/fake)
IMG_W = 96            # Image width for training
IMG_H = 96            # Image height for training
BATCH_SIZE = 16       # Batch size
MAX_STEP = 4000       # Maximum number of training steps
LEARNING_RATE = 0.0001  # Learning rate

# Directories
TRAIN_DIR = "/content/train_data/"
LOGS_TRAIN_DIR = "/content/model_checkpoints/"

# Create directories if they don't exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(LOGS_TRAIN_DIR, exist_ok=True)

print("Training parameters configured:")
print(f"Image dimensions: {IMG_W}x{IMG_H}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum steps: {MAX_STEP}")
print(f"Learning rate: {LEARNING_RATE}")
```

3. **Define Functions** (Third Code Cell):

```python
# Copy SECTIONS 3, 4, 5, 6 from train_model_colab_script.py
# This includes all the functions for data processing, model architecture,
# training, and helper functions

# For brevity, these sections aren't shown here.
# Copy all the function definitions from the script.
```

4. **Mount Google Drive** (Optional):

```python
# Mount Google Drive if your data is stored there
drive.mount('/content/drive')
print("Google Drive mounted at /content/drive")
```

5. **Prepare Training Data**:

There are two options:

**Option 1: Upload a zip file containing extracted frames**
```python
# Upload and extract training data from a zip file
print("Please upload a zip file containing your training data (images labeled as real.X.png and fake.X.png)")
uploaded = files.upload()  # Upload zip file

if list(uploaded.keys()):
    zip_filename = list(uploaded.keys())[0]
    
    # Extract the uploaded zip file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(TRAIN_DIR)
    
    print(f"Extracted {zip_filename} to {TRAIN_DIR}")
    
    # Count real and fake images
    real_images = glob.glob(os.path.join(TRAIN_DIR, "real*.png"))
    fake_images = glob.glob(os.path.join(TRAIN_DIR, "fake*.png"))
    
    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
```

**Option 2: Copy from Google Drive**
```python
# Copy training data from Google Drive
drive_path = "/content/drive/MyDrive/your_folder_with_frames"  # Change this to your path

# Copy files from Drive to the training directory
!cp -r {drive_path}/* {TRAIN_DIR}

# Count real and fake images
real_images = glob.glob(os.path.join(TRAIN_DIR, "real*.png"))
fake_images = glob.glob(os.path.join(TRAIN_DIR, "fake*.png"))

print(f"Copied from Google Drive: {len(real_images)} real images and {len(fake_images)} fake images")
```

6. **Train the Model**:

```python
# Train the model
model, history = run_training(TRAIN_DIR, LOGS_TRAIN_DIR)
print("\nTraining complete!")
```

7. **Plot Training History**:

```python
# Plot the training history
plot_training_history(history)
```

8. **Download the Trained Model**:

```python
# Download the model
final_model_path = os.path.join(LOGS_TRAIN_DIR, 'final_model.h5')
files.download(final_model_path)
print(f"Downloaded {final_model_path}")
```

9. **Test the Model** (Optional):

```python
# Test the model on some sample images
test_model_on_samples(model)
```

### 5. Common Issues and Solutions

1. **Out of Memory (OOM) Errors**:
   - Reduce the `BATCH_SIZE` (e.g., 8 instead of 16)
   - Use smaller image dimensions (e.g., 64x64 instead of 96x96)
   - Restart the runtime to clear memory (Runtime → Restart runtime)

2. **Slow Training**:
   - Make sure GPU acceleration is enabled
   - Reduce image dimensions if needed

3. **Model Not Improving**:
   - Check that your training data is correctly labeled
   - Try adjusting the learning rate (e.g., 0.0005 or 0.00005 instead of 0.0001)
   - Train for more epochs by increasing `MAX_STEP`

4. **Google Drive Mount Expiration**:
   - If your Google Drive mount expires during a long training session, you'll need to remount it and restart from a checkpoint

### 6. Additional Tips

1. **Save Checkpoints Frequently**:
   - The script already saves checkpoints after each epoch
   - You can download these checkpoints to resume training later

2. **Data Augmentation** (Advanced):
   - For better performance, you might want to add data augmentation to the training
   - This would involve modifying the `create_dataset` function

3. **Monitoring Training**:
   - The script displays the training progress
   - You can also use TensorBoard for more detailed monitoring

4. **Testing with New Images**:
   - After training, you can test the model with new images by using the `test_model_on_samples` function
   - Just replace the test images with your own test set

5. **Save Your Notebook**:
   - Once your implementation is working, save a copy of the notebook for future use
   - You can also export it to GitHub or download it to your computer

## Conclusion

You now have a consolidated script to train your face anti-spoofing model in Google Colab. This approach puts all the necessary code in one file, making it easier to implement and modify. The script includes all the functionality of the original multi-file approach, with added conveniences for Google Colab usage. 