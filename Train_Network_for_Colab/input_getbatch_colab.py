# This script is used to preprocess and prepare image data 
import tensorflow as tf
import numpy as np
import os

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