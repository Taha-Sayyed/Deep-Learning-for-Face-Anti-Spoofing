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
%%writefile main_dropout_kaggle_fixed.py
import os
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from packaging import version

# Import our custom modules
from model_c_kaggle import create_model, compile_model, create_ensemble_model
from input_getbatch_kaggle import load_kaggle_dataset

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Face Anti-Spoofing Model on Kaggle')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/face-anti-spoofing-dataset',
                        help='Directory containing the dataset')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size (both height and width)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    # Model parameters
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate for regularization')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes (2 for binary classification)')
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble model architecture')
    parser.add_argument('--num_models', type=int, default=3,
                        help='Number of models in the ensemble')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use data augmentation during training')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='/kaggle/working',
                        help='Directory to save models and results')
    parser.add_argument('--model_name', type=str, default='face_anti_spoofing',
                        help='Base name for saved models')
    
    args = parser.parse_args()
    return args

def create_callbacks(output_dir, model_name, patience=10):
    """Create callbacks for model training"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current timestamp for unique model names
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Check TensorFlow version to determine the appropriate file extension
    # TensorFlow 2.13+ uses .keras, older versions use .h5
    if version.parse(tf.__version__) >= version.parse("2.13.0"):
        file_ext = ".keras"
        print("Using .keras file extension (TensorFlow >= 2.13)")
    else:
        file_ext = ".h5"
        print("Using .h5 file extension (TensorFlow < 2.13)")
    
    model_path = os.path.join(output_dir, f"{model_name}_{timestamp}{file_ext}")
    checkpoint_path = os.path.join(output_dir, f"model_{{epoch:02d}}-{{val_loss:.4f}}{file_ext}")
    log_dir = os.path.join(output_dir, "logs", timestamp)
    
    callbacks = [
        # Save the best model based on validation loss
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Save checkpoints during training
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=False,
            save_freq='epoch',
            verbose=0
        ),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    
    # Add early stopping if requested
    if patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
        )
    
    return callbacks, model_path

def plot_training_history(history, output_dir, model_name, timestamp=None):
    """Plot and save training history metrics"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training and validation accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_history_{timestamp}.png"))
    
    # Save history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(output_dir, f"{model_name}_history_{timestamp}.csv"), index=False)
    
    plt.close()

def evaluate_model(model, test_dataset):
    """Evaluate the model on the test dataset"""
    print("\nEvaluating model on test dataset...")
    results = model.evaluate(test_dataset)
    
    metrics = model.metrics_names
    for metric, value in zip(metrics, results):
        print(f"{metric}: {value:.4f}")
    
    return {metric: value for metric, value in zip(metrics, results)}

def save_predictions(model, test_dataset, output_dir, model_name, timestamp=None):
    """Generate and save predictions on test data"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Get predictions
    predictions = model.predict(test_dataset)
    
    # Get true labels from the test dataset
    true_labels = np.concatenate([y for _, y in test_dataset], axis=0)
    
    # For binary classification, get predicted class labels
    if predictions.shape[1] == 1:  # Binary with sigmoid
        pred_classes = (predictions > 0.5).astype(int).flatten()
        pred_probs = predictions.flatten()
    else:  # Multi-class with softmax
        pred_classes = np.argmax(predictions, axis=1)
        pred_probs = np.max(predictions, axis=1)
    
    # Create DataFrame for predictions
    results_df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_class': pred_classes,
        'confidence': pred_probs
    })
    
    # Save predictions to CSV
    results_path = os.path.join(output_dir, f"{model_name}_predictions_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")
    
    return results_df

def train_model(args):
    """Train and evaluate the anti-spoofing model"""
    print("Starting Face Anti-Spoofing Training")
    print(f"Training parameters: {args}")
    
    # Set up the timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Load dataset
    print("\nLoading dataset...")
    train_generator, val_generator, test_dataset, train_steps, val_steps = load_kaggle_dataset(
        args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        use_augmentation=args.use_augmentation
    )
    
    # Create model
    print("\nCreating model...")
    if args.ensemble:
        model = create_ensemble_model(
            input_shape=(args.img_size, args.img_size, 3),
            num_classes=args.num_classes,
            dropout_rate=args.dropout_rate,
            num_models=args.num_models
        )
    else:
        model = create_model(
            input_shape=(args.img_size, args.img_size, 3),
            num_classes=args.num_classes,
            dropout_rate=args.dropout_rate
        )
    
    # Compile the model
    model = compile_model(model, args.num_classes, lr=args.learning_rate)
    model.summary()
    
    # Create callbacks
    callbacks, model_path = create_callbacks(
        args.output_dir,
        args.model_name,
        patience=args.patience if args.early_stopping else 0
    )
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    try:
        # Try to load the best model for evaluation
        print(f"\nLoading best model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        # Evaluate model on test dataset
        test_results = evaluate_model(model, test_dataset)
        
        # Plot and save training history
        plot_training_history(history, args.output_dir, args.model_name, timestamp)
        
        # Save predictions
        save_predictions(model, test_dataset, args.output_dir, args.model_name, timestamp)
        
        # Save the test results
        test_results_path = os.path.join(args.output_dir, f"{args.model_name}_test_results_{timestamp}.csv")
        pd.DataFrame([test_results]).to_csv(test_results_path, index=False)
        
        print(f"\nTraining complete. Model saved to {model_path}")
    except Exception as e:
        print(f"Error in evaluation phase: {str(e)}")
        print("Continuing with the trained model from the last epoch")
        test_results = None
    
    return model, test_results, history

if __name__ == "__main__":
    # Get the TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Make sure packaging is installed for version comparison
    try:
        import packaging.version
    except ImportError:
        print("Installing packaging module for version comparison...")
        import pip
        pip.main(['install', 'packaging'])
        import packaging.version
    
    # Configure GPU memory growth if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    
    # Parse command line arguments
    args = parse_args()
    
    # Train the model
    model, test_results, history = train_model(args)
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