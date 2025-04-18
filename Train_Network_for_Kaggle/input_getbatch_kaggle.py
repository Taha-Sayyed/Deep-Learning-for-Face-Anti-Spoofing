import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import glob

def load_dataset(data_dir, img_size=(128, 128), test_size=0.2, val_size=0.1, random_state=42):
    """
    Load face anti-spoofing dataset from directory structure.
    
    Args:
        data_dir (str): Directory containing the dataset
        img_size (tuple): Image dimensions (height, width)
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_images, train_labels, val_images, val_labels, test_images, test_labels)
    """
    print("Loading dataset from:", data_dir)
    
    # Get real and fake image paths
    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    real_images = glob.glob(os.path.join(real_dir, '*.png')) + glob.glob(os.path.join(real_dir, '*.jpg'))
    fake_images = glob.glob(os.path.join(fake_dir, '*.png')) + glob.glob(os.path.join(fake_dir, '*.jpg'))
    
    print(f"Found {len(real_images)} real images and {len(fake_images)} fake images")
    
    all_image_paths = real_images + fake_images
    all_labels = [1] * len(real_images) + [0] * len(fake_images)  # 1 for real, 0 for fake
    
    # Split into train+val and test sets
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_image_paths, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    
    # Split train+val into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=train_val_labels
    )
    
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")
    
    # Load and preprocess the images
    train_images = load_and_preprocess_images(train_paths, img_size)
    val_images = load_and_preprocess_images(val_paths, img_size)
    test_images = load_and_preprocess_images(test_paths, img_size)
    
    return (
        train_images, np.array(train_labels), 
        val_images, np.array(val_labels), 
        test_images, np.array(test_labels)
    )

def load_and_preprocess_images(image_paths, img_size):
    """
    Load and preprocess images from paths.
    
    Args:
        image_paths (list): List of image file paths
        img_size (tuple): Target image size (height, width)
        
    Returns:
        np.ndarray: Array of preprocessed images
    """
    images = []
    
    for img_path in tqdm(image_paths, desc="Loading images"):
        try:
            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Resize image
            img = cv2.resize(img, (img_size[1], img_size[0]))
            
            # Preprocess image
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    return np.array(images)

def create_data_generators(train_images, train_labels, val_images, val_labels, 
                           batch_size=32, use_augmentation=True):
    """
    Create data generators for training and validation.
    
    Args:
        train_images (np.ndarray): Training images
        train_labels (np.ndarray): Training labels
        val_images (np.ndarray): Validation images
        val_labels (np.ndarray): Validation labels
        batch_size (int): Batch size
        use_augmentation (bool): Whether to use data augmentation
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    if use_augmentation:
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator()
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        train_images, train_labels,
        batch_size=batch_size,
        shuffle=True
    )
    
    validation_generator = val_datagen.flow(
        val_images, val_labels,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, validation_generator

def create_tf_dataset(images, labels, batch_size=32, shuffle=True, 
                      use_augmentation=False, is_training=True):
    """
    Create a TensorFlow Dataset for training or evaluation.
    
    Args:
        images (np.ndarray): Input images
        labels (np.ndarray): Target labels
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        use_augmentation (bool): Whether to use data augmentation
        is_training (bool): Whether the dataset is for training
        
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    # Shuffle if needed
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    
    # Apply data augmentation if needed
    if use_augmentation and is_training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def augment_image(image, label):
    """
    Apply data augmentation to an image using TensorFlow operations.
    
    Args:
        image: Input image tensor
        label: Input label tensor
        
    Returns:
        tuple: (augmented_image, label)
    """
    # Random flip
    image = tf.image.random_flip_left_right(image)
    
    # Random brightness and contrast
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    # Random saturation and hue
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)
    
    # Ensure values are in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

def load_kaggle_dataset(data_dir, img_size=(128, 128), batch_size=32, 
                       use_augmentation=True, test_size=0.2, val_size=0.1):
    """
    Load the dataset and prepare it for training on Kaggle.
    
    Args:
        data_dir (str): Directory containing the dataset
        img_size (tuple): Image dimensions (height, width)
        batch_size (int): Batch size
        use_augmentation (bool): Whether to use data augmentation
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        
    Returns:
        tuple: (train_generator, validation_generator, test_dataset, 
               train_steps_per_epoch, val_steps_per_epoch)
    """
    # Load the dataset
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_dataset(
        data_dir, img_size, test_size, val_size
    )
    
    # Create data generators for training and validation
    train_generator, validation_generator = create_data_generators(
        train_images, train_labels, val_images, val_labels, 
        batch_size, use_augmentation
    )
    
    # Create a TensorFlow Dataset for testing
    test_dataset = create_tf_dataset(
        test_images, test_labels, batch_size, 
        shuffle=False, use_augmentation=False, is_training=False
    )
    
    # Calculate steps per epoch
    train_steps_per_epoch = len(train_images) // batch_size
    val_steps_per_epoch = len(val_images) // batch_size
    
    return (
        train_generator, validation_generator, test_dataset,
        train_steps_per_epoch, val_steps_per_epoch
    ) 