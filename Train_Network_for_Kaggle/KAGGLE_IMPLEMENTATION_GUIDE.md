# Kaggle Implementation Guide for Face Anti-Spoofing Model Training

This guide will walk you through the process of training a deep learning model for face anti-spoofing using Kaggle notebooks. Kaggle provides free access to GPUs and TPUs, making it an excellent alternative to Google Colab for training deep learning models.

## Contents

1. [Getting Started with Kaggle](#getting-started-with-kaggle)
2. [Setting Up Your Kaggle Notebook](#setting-up-your-kaggle-notebook)
3. [Preparing Your Dataset](#preparing-your-dataset)
4. [Running the Training Script](#running-the-training-script)
5. [Optimizing Performance](#optimizing-performance)
6. [Troubleshooting](#troubleshooting)
7. [Saving and Downloading Results](#saving-and-downloading-results)

## Getting Started with Kaggle

If you're new to Kaggle, here's how to get started:

1. **Create a Kaggle Account**:
   - Go to [Kaggle.com](https://www.kaggle.com/) and sign up for a free account.
   - Verify your account through the email link sent to you.

2. **Explore Kaggle's Features**:
   - Kaggle offers datasets, competitions, notebooks, and more.
   - For this project, we'll be using Kaggle Notebooks (previously called Kernels).

3. **Understand Kaggle's Free GPU Quotas**:
   - Kaggle provides 30 hours of GPU and 20 hours of TPU usage per week for free.
   - This is more consistent than Google Colab's free tier, which often disconnects.

## Setting Up Your Kaggle Notebook

Follow these steps to set up your Kaggle notebook for face anti-spoofing model training:

1. **Create a New Notebook**:
   - From the Kaggle homepage, click on "Create" at the top and select "Notebook".
   - This will open a new Jupyter notebook in your browser.

2. **Configure Notebook Settings**:
   - Click on the settings icon (⚙️) on the right side of the page.
   - Under "Accelerator", select "GPU" for faster training.
   - Set "Language" to "Python".
   - Click "Save" to apply these settings.

3. **Understanding Kaggle's Directory Structure**:
   - `/kaggle/input/`: This is where your datasets will be mounted.
   - `/kaggle/working/`: This is your working directory where outputs are saved.

4. **Upload the Training Script**:
   - In your notebook, create a new code cell.
   - Copy and paste the following code to create the training script file:

```python
%%writefile train_model_kaggle_script.py
# Paste the entire content of train_model_kaggle_script.py here
```

   - Run this cell to create the script file in your working directory.

## Preparing Your Dataset

Before training, you need to prepare your dataset. There are two ways to do this:

### Option 1: Upload Your Own Dataset

1. **Create a New Dataset**:
   - Click on "Create" at the top of the Kaggle page and select "Dataset".
   - Give your dataset a title and description.
   - Upload your dataset files (make sure your images follow the naming convention: `real.X.png` and `fake.X.png`).
   - Optionally, you can upload a ZIP file containing all your images.
   - Click "Create" to create your dataset.

2. **Add the Dataset to Your Notebook**:
   - Go back to your notebook.
   - Click on the "Add Data" button on the right sidebar.
   - Find your dataset and click "Add" to mount it to your notebook.

### Option 2: Use an Existing Dataset

1. **Find an Appropriate Dataset**:
   - Go to [Kaggle Datasets](https://www.kaggle.com/datasets) and search for face anti-spoofing datasets.
   - If you've already extracted frames using the Video cutframe tool, you can upload those as a dataset.

2. **Add the Dataset to Your Notebook**:
   - Once you've found a suitable dataset, click on "Add" to mount it to your notebook.
   - The dataset will be available in the `/kaggle/input/` directory.

## Running the Training Script

You have two options for running the training script:

### Option 1: Run the Full Script Directly

Add this to a code cell in your notebook:

```python
%run train_model_kaggle_script.py
```

This will run the entire script and prompt you to select from available datasets and confirm training steps.

### Option 2: Run the Script Section by Section (Recommended)

This approach gives you more control over each step. Copy and run each section from the script:

1. **Setup Environment**:

```python
# Import required libraries
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import zipfile
import glob
import shutil
from pathlib import Path

# Check TensorFlow version and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
```

2. **Configure Parameters**:

```python
# Training parameters - Feel free to modify these
N_CLASSES = 2         # Number of classes (real/fake)
IMG_W = 96            # Image width for training
IMG_H = 96            # Image height for training
BATCH_SIZE = 16       # Batch size
MAX_STEP = 4000       # Maximum number of training steps
LEARNING_RATE = 0.0001  # Learning rate

# Directories - Kaggle specific paths
# Input data is in /kaggle/input
# Output should be saved to /kaggle/working
TRAIN_DIR = "/kaggle/working/train_data/"
LOGS_TRAIN_DIR = "/kaggle/working/model_checkpoints/"

# Create directories if they don't exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(LOGS_TRAIN_DIR, exist_ok=True)

print("Training parameters configured:")
print(f"Image dimensions: {IMG_W}x{IMG_H}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum steps: {MAX_STEP}")
print(f"Learning rate: {LEARNING_RATE}")
```

3. **Define All Functions**:
   - Copy and paste all the functions from `train_model_kaggle_script.py` into a new cell.
   - This includes `get_files()`, `create_dataset()`, `create_model()`, etc.

4. **List Available Datasets**:

```python
# List available datasets
input_dir = "/kaggle/input"
dataset_paths = []
for item in os.listdir(input_dir):
    item_path = os.path.join(input_dir, item)
    if os.path.isdir(item_path):
        dataset_paths.append(item_path)

print(f"Found {len(dataset_paths)} datasets:")
for i, path in enumerate(dataset_paths):
    print(f"{i+1}. {path}")
```

5. **Prepare Training Data**:

```python
# Select the dataset containing training images
# Replace this with the actual path to your dataset
selected_dataset = dataset_paths[0]  # Choose the appropriate dataset
print(f"Selected dataset: {selected_dataset}")

# Check if dataset contains a zip file
zip_files = glob.glob(os.path.join(selected_dataset, "*.zip"))
if zip_files:
    print(f"Found zip file(s): {zip_files}")
    zip_path = zip_files[0]  # Use the first zip file
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TRAIN_DIR)
    
    print(f"Extracted {zip_path} to {TRAIN_DIR}")
else:
    # Look for PNG files directly
    png_files = glob.glob(os.path.join(selected_dataset, "*.png"))
    if png_files:
        for png_file in png_files:
            shutil.copy(png_file, TRAIN_DIR)
        print(f"Copied {len(png_files)} PNG files to {TRAIN_DIR}")
    else:
        # Look for PNG files in subdirectories
        for subdir in os.listdir(selected_dataset):
            subdir_path = os.path.join(selected_dataset, subdir)
            if os.path.isdir(subdir_path):
                png_files = glob.glob(os.path.join(subdir_path, "*.png"))
                if png_files:
                    for png_file in png_files:
                        shutil.copy(png_file, TRAIN_DIR)
                    print(f"Copied {len(png_files)} PNG files from {subdir} to {TRAIN_DIR}")

# Count real and fake images
real_images = glob.glob(os.path.join(TRAIN_DIR, "real*.png"))
fake_images = glob.glob(os.path.join(TRAIN_DIR, "fake*.png"))
print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
```

6. **Train the Model**:

```python
# Train the model
model, history = run_training(TRAIN_DIR, LOGS_TRAIN_DIR)
print("\nTraining complete!")
```

7. **Plot Training History and Test the Model**:

```python
# Plot the training history
plot_training_history(history)

# Test the model on sample images
test_model_on_samples(model)
```

## Optimizing Performance

To get the best performance on Kaggle:

1. **Adjust Batch Size**: 
   - If you encounter Out of Memory (OOM) errors, reduce the batch size (e.g., 8 instead of 16).
   - If training is slow, try increasing the batch size if memory allows.

2. **Image Dimensions**: 
   - Smaller images (e.g., 64x64) train faster but may lose some detail.
   - Larger images (e.g., 128x128) provide more detail but require more memory and time.

3. **Learning Rate**: 
   - If training is unstable, try a smaller learning rate (e.g., 0.00005).
   - If training is too slow, try a larger learning rate (e.g., 0.0005).

4. **Maximum Steps**: 
   - The default 4000 steps may be more than needed. Monitor the accuracy and stop early if it plateaus.

5. **Use the GPU Accelerator**: 
   - Always ensure you've selected "GPU" in the notebook settings.

## Troubleshooting

Here are some common issues and their solutions:

1. **Out of Memory Errors**:
   - Reduce batch size
   - Use smaller image dimensions
   - Simplify model architecture if necessary

2. **Dataset Not Found**:
   - Ensure you've added the dataset to your notebook using the "Add Data" button
   - Check the path in `/kaggle/input/` to confirm it's mounted

3. **No PNG Files Found**:
   - Ensure your dataset contains images with the correct naming format (`real.X.png` and `fake.X.png`)
   - Check if images are in a subdirectory within your dataset

4. **Training Not Improving**:
   - Ensure your dataset has a good balance of real and fake images
   - Try adjusting the learning rate
   - Check if your images are properly labeled

## Saving and Downloading Results

Kaggle automatically saves files created in the `/kaggle/working/` directory. To download your trained model:

1. **Find Your Files**:
   - After training completes, look for the "Output" tab in your notebook.
   - You should see `final_model.h5` and other saved models there.

2. **Download the Model**:
   - Click on the download icon next to each file you want to download.
   - Alternatively, you can create a new dataset from your outputs for future use.

3. **Create a Dataset from Outputs** (Optional):
   - Click on the "Data" tab in your notebook.
   - Click on "Create" next to "Output Product".
   - Name your dataset and click "Create".
   - This creates a dataset containing all files from your `/kaggle/working/` directory.

## Conclusion

Training a face anti-spoofing model on Kaggle provides several advantages over Google Colab, including more reliable GPU access and longer runtime sessions. By following this guide, you should be able to successfully train your model and achieve good performance.

Remember that machine learning is an iterative process. Don't be afraid to experiment with different parameters and approaches to improve your model's performance. 