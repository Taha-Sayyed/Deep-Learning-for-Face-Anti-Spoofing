#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Anti-Spoofing Model Training for Google Colab

This script trains a deep learning model to distinguish between real and fake faces.
It's designed to be run in Google Colab cells, section by section.

Author: [Your Name]
"""

###############################################################################
# SECTION 1: Setup Environment and Required Libraries
###############################################################################

# Import required libraries
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

###############################################################################
# SECTION 2: Define Constants and Parameters
###############################################################################

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

###############################################################################
# SECTION 3: Define Input Processing Functions
###############################################################################

def get_files(file_dir):
    """Get all image files and their labels from the given directory
    
    Args:
        file_dir: directory containing image files
        
    Returns:
        all_image_list: list of all image file paths
        all_label_list: list of corresponding labels
    """
    real = []
    label_real = []
    fake = []
    label_fake = []

    for file in os.listdir(file_dir):
        name = file.split(sep='.')        
        # Divide images named like "fake.1.png" into real or fake groups 
        
        if name[0]=='real':
            real.append(file_dir + '/' + file)  
            label_real.append(1)          
        if name[0]=='fake':
            fake.append(file_dir + '/' + file)
            label_fake.append(0)
    print('There are %d real images\nThere are %d fake images' %(len(real), len(fake)))
    
    image_list = np.hstack((real, fake))
    label_list = np.hstack((label_real, label_fake))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()  
    np.random.shuffle(temp)  
    
    all_image_list = temp[:, 0]  
    all_label_list = temp[:, 1]   
    all_label_list = [int(float(i)) for i in all_label_list]

    return all_image_list, all_label_list

def load_and_preprocess_image(image_path, image_size):
    """Load and preprocess a single image
    
    Args:
        image_path: path to the image file
        image_size: tuple of (width, height)
        
    Returns:
        processed_image: processed image tensor
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.image.per_image_standardization(img)
    return img

def create_dataset(image_list, label_list, image_size, batch_size, shuffle=True):
    """Create a TensorFlow dataset from image paths and labels
    
    Args:
        image_list: list of image file paths
        label_list: list of corresponding labels
        image_size: tuple of (width, height)
        batch_size: batch size for training
        shuffle: whether to shuffle the dataset
        
    Returns:
        dataset: TensorFlow dataset containing image batches and labels
    """
    # Convert lists to tensors
    image_paths = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    
    # Create a dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Map the preprocessing function to each element
    dataset = dataset.map(
        lambda x, y: (load_and_preprocess_image(x, image_size), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Shuffle and batch the dataset
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_list))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

###############################################################################
# SECTION 4: Define Model Architecture
###############################################################################

def create_model(input_shape, n_classes=2):
    """Create a CNN model for face anti-spoofing, based on the original TF1.x model
    
    Args:
        input_shape: tuple of input shape (height, width, channels)
        n_classes: number of output classes
        
    Returns:
        model: Keras model
    """
    # Use Keras functional API to match the original model structure
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Block 1: Conv1 -> Pool1
    x = tf.keras.layers.Conv2D(64, (7, 7), padding='same', 
                     kernel_initializer=tf.keras.initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv1')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)
    x = tf.keras.layers.Dropout(0.5, name='pool1_dropout')(x)
    
    # Block 2: Conv2 -> Pool2
    x = tf.keras.layers.Conv2D(128, (5, 5), padding='same',
                     kernel_initializer=tf.keras.initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)
    x = tf.keras.layers.Dropout(0.5, name='pool2_dropout')(x)
    
    # Block 3: Conv3_1 -> Conv3_2 -> Pool3
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same',
                     kernel_initializer=tf.keras.initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv3_1')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same',
                     kernel_initializer=tf.keras.initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv3_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='pool3')(x)
    x = tf.keras.layers.Dropout(0.5, name='pool3_dropout')(x)
    
    # Block 4: Conv4_1 -> Conv4_2 -> Pool4
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same',
                     kernel_initializer=tf.keras.initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv4_1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same',
                     kernel_initializer=tf.keras.initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv4_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='pool4')(x)
    x = tf.keras.layers.Dropout(0.5, name='pool4_dropout')(x)
    
    # Block 5: Conv5_1 -> Conv5_2 -> Pool5
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same',
                     kernel_initializer=tf.keras.initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv5_1')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same',
                     kernel_initializer=tf.keras.initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv5_2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='pool5')(x)
    x = tf.keras.layers.Dropout(0.5, name='pool5_dropout')(x)
    
    # Fully connected layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, 
                    kernel_regularizer=tf.keras.regularizers.l2(0.005),
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    activation='relu', name='local3')(x)
    x = tf.keras.layers.Dropout(0.5, name='local3_dropout')(x)
    
    x = tf.keras.layers.Dense(128, 
                    kernel_regularizer=tf.keras.regularizers.l2(0.005),
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    activation='relu', name='local4')(x)
    x = tf.keras.layers.Dropout(0.5, name='local4_dropout')(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(n_classes, 
                          kernel_regularizer=tf.keras.regularizers.l2(0.005),
                          kernel_initializer=tf.keras.initializers.VarianceScaling(
                              scale=1.0, mode='fan_in', distribution='uniform'),
                          bias_initializer=tf.keras.initializers.Constant(0.1),
                          name='softmax_linear')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='face_antispoofing_model')
    return model

def compile_model(model, learning_rate=0.0001):
    """Compile the model with appropriate loss, optimizer and metrics
    
    Args:
        model: Keras model to compile
        learning_rate: learning rate for the optimizer
        
    Returns:
        model: Compiled Keras model
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Use sparse categorical crossentropy since we have integer labels
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

###############################################################################
# SECTION 5: Define Training Functions
###############################################################################

def plot_training_history(history):
    """Plot the training history (accuracy and loss)
    
    Args:
        history: training history from model.fit()
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.show()

def run_training(train_dir, logs_train_dir):
    """Run the training process for the face anti-spoofing model
    
    Args:
        train_dir: directory containing training images
        logs_train_dir: directory to save model checkpoints
    """
    # Create directory for model checkpoints if it doesn't exist
    os.makedirs(logs_train_dir, exist_ok=True)
    
    # Get files and labels
    image_list, label_list = get_files(train_dir)
    
    # Create TF dataset
    train_dataset = create_dataset(
        image_list, 
        label_list,
        (IMG_W, IMG_H),
        BATCH_SIZE
    )
    
    # Create and compile the model
    model = create_model(input_shape=(IMG_W, IMG_H, 3), n_classes=N_CLASSES)
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    # Display model summary
    model.summary()
    
    # Create callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs_train_dir, 
        histogram_freq=1
    )
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(logs_train_dir, 'model.{epoch:02d}-{accuracy:.2f}.h5'),
        save_weights_only=False,
        save_best_only=True,
        monitor='accuracy',
        save_freq='epoch',
        verbose=1
    )
    
    # Train the model
    steps_per_epoch = len(image_list) // BATCH_SIZE
    epochs = MAX_STEP // steps_per_epoch  # Convert steps to epochs
    print(f"Training for {epochs} epochs with {steps_per_epoch} steps per epoch")
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tensorboard_callback, checkpoint_callback]
    )
    
    # Save the final model
    final_model_path = os.path.join(logs_train_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save in SavedModel format (TF 2.x preferred format)
    saved_model_path = os.path.join(logs_train_dir, 'saved_model')
    model.save(saved_model_path)
    print(f"SavedModel saved to {saved_model_path}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

###############################################################################
# SECTION 6: Helper Functions for Data Preparation
###############################################################################

def mount_google_drive():
    """Mount Google Drive to access files"""
    drive.mount('/content/drive')
    print("Google Drive mounted at /content/drive")

def upload_python_files():
    """Upload the Python modules from local system"""
    print("Please upload the Python files (if you're not using this script directly)")
    uploaded = files.upload()
    return list(uploaded.keys())

def upload_and_extract_training_data():
    """Upload a zip file containing training data and extract it"""
    print("Please upload a zip file containing your training data (images labeled as real.X.png and fake.X.png)")
    uploaded = files.upload()  # Upload zip file
    
    if not list(uploaded.keys()):
        print("No file was uploaded. Please try again.")
        return False
    
    zip_filename = list(uploaded.keys())[0]
    
    # Check if it's a zip file
    if not zip_filename.lower().endswith('.zip'):
        print(f"Uploaded file {zip_filename} is not a zip file.")
        return False
    
    # Extract the uploaded zip file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(TRAIN_DIR)
    
    print(f"Extracted {zip_filename} to {TRAIN_DIR}")
    
    # Count real and fake images
    real_images = glob.glob(os.path.join(TRAIN_DIR, "real*.png"))
    fake_images = glob.glob(os.path.join(TRAIN_DIR, "fake*.png"))
    
    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
    return True

def copy_from_drive(drive_path):
    """Copy training data from Google Drive to Colab instance"""
    if not os.path.exists('/content/drive'):
        print("Google Drive is not mounted. Please mount it first.")
        return False
    
    if not os.path.exists(drive_path):
        print(f"Path {drive_path} not found in Google Drive.")
        return False
    
    # Copy files from Drive to the training directory
    command = f"cp -r {drive_path}/* {TRAIN_DIR}"
    os.system(command)
    
    # Count real and fake images
    real_images = glob.glob(os.path.join(TRAIN_DIR, "real*.png"))
    fake_images = glob.glob(os.path.join(TRAIN_DIR, "fake*.png"))
    
    print(f"Copied from Google Drive: {len(real_images)} real images and {len(fake_images)} fake images")
    return True

def download_model():
    """Download the trained model file"""
    final_model_path = os.path.join(LOGS_TRAIN_DIR, 'final_model.h5')
    if os.path.exists(final_model_path):
        files.download(final_model_path)
        print(f"Downloaded {final_model_path}")
    else:
        print(f"Model file {final_model_path} not found.")

def zip_and_download_checkpoints():
    """Zip all checkpoints and download them"""
    if not os.path.exists(LOGS_TRAIN_DIR):
        print(f"Checkpoints directory {LOGS_TRAIN_DIR} not found.")
        return
    
    zip_path = "/content/model_checkpoints.zip"
    
    # Create zip file
    command = f"zip -r {zip_path} {LOGS_TRAIN_DIR}"
    os.system(command)
    
    # Download the zip file
    if os.path.exists(zip_path):
        files.download(zip_path)
        print(f"Downloaded checkpoints as {zip_path}")
    else:
        print(f"Failed to create zip file {zip_path}")

def test_model_on_samples(model):
    """Test the trained model on some sample images
    
    Args:
        model: trained Keras model
    """
    # Get all image paths
    image_list, label_list = get_files(TRAIN_DIR)
    
    # Split into real and fake
    real_samples = [path for path in image_list if 'real' in path]
    fake_samples = [path for path in image_list if 'fake' in path]
    
    # If we don't have any samples, return
    if not real_samples or not fake_samples:
        print("Not enough samples to test the model.")
        return
    
    # Select a few random samples from each class
    test_real = random.sample(real_samples, min(3, len(real_samples)))
    test_fake = random.sample(fake_samples, min(3, len(fake_samples)))
    test_images = test_real + test_fake
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    class_names = ['Fake', 'Real']
    
    # Test each image
    for i, image_path in enumerate(test_images):
        # Preprocess the image
        img_tensor = load_and_preprocess_image(image_path, (IMG_W, IMG_H))
        img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension
        
        # Make prediction
        predictions = model(img_tensor)
        score = tf.nn.softmax(predictions[0])
        
        # Get class and confidence
        class_idx = np.argmax(score)
        confidence = 100 * np.max(score)
        
        # Load the original image for display
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        
        # Display the image and prediction
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        true_class = 1 if 'real' in image_path else 0
        color = 'green' if class_idx == true_class else 'red'
        plt.title(f"Pred: {class_names[class_idx]} ({confidence:.1f}%)\nTrue: {class_names[true_class]}", color=color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

###############################################################################
# SECTION 7: Main Function
###############################################################################

def main():
    """Main function to coordinate the training process"""
    print("===== Face Anti-Spoofing Model Training =====")
    
    # 1. Check if we need to mount Google Drive
    print("\nDo you want to mount Google Drive? (y/n)")
    if input().lower() == 'y':
        mount_google_drive()
    
    # 2. Get training data
    print("\nHow do you want to provide training data?")
    print("1. Upload a zip file")
    print("2. Copy from Google Drive")
    choice = input("Enter 1 or 2: ")
    
    if choice == '1':
        if not upload_and_extract_training_data():
            print("Failed to upload and extract training data. Exiting.")
            return
    elif choice == '2':
        print("Enter the path to your training data in Google Drive (e.g., /content/drive/MyDrive/frames):")
        drive_path = input()
        if not copy_from_drive(drive_path):
            print("Failed to copy training data from Google Drive. Exiting.")
            return
    else:
        print("Invalid choice. Exiting.")
        return
    
    # 3. Run training
    print("\nDo you want to start training now? (y/n)")
    if input().lower() == 'y':
        model, history = run_training(TRAIN_DIR, LOGS_TRAIN_DIR)
        
        # 4. Download the model
        print("\nDo you want to download the trained model? (y/n)")
        if input().lower() == 'y':
            download_model()
        
        # 5. Download checkpoints
        print("\nDo you want to download all checkpoints as a zip file? (y/n)")
        if input().lower() == 'y':
            zip_and_download_checkpoints()
        
        # 6. Test the model
        print("\nDo you want to test the model on some sample images? (y/n)")
        if input().lower() == 'y':
            test_model_on_samples(model)
    
    print("\n===== Process Complete =====")

if __name__ == "__main__":
    main() 