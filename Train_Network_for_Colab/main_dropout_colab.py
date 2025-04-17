import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import input_getbatch
import model_c

# Constants
N_CLASSES = 2
IMG_W = 96  
IMG_H = 96
BATCH_SIZE = 16
MAX_STEP = 4000 
LEARNING_RATE = 0.0001

def run_training(train_dir, logs_train_dir):
    """Run the training process for the face anti-spoofing model
    
    Args:
        train_dir: directory containing training images
        logs_train_dir: directory to save model checkpoints
    """
    # Create directory for model checkpoints if it doesn't exist
    os.makedirs(logs_train_dir, exist_ok=True)
    
    # Get files and labels
    image_list, label_list = input_getbatch.get_files(train_dir)
    
    # Create TF dataset
    train_dataset = input_getbatch.create_dataset(
        image_list, 
        label_list,
        (IMG_W, IMG_H),
        BATCH_SIZE
    )
    
    # Create and compile the model
    model = model_c.create_model(input_shape=(IMG_W, IMG_H, 3), n_classes=N_CLASSES)
    model = model_c.compile_model(model, learning_rate=LEARNING_RATE)
    
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
    history = model.fit(
        train_dataset,
        epochs=MAX_STEP // steps_per_epoch,  # Convert steps to epochs
        steps_per_epoch=steps_per_epoch,
        callbacks=[tensorboard_callback, checkpoint_callback]
    )
    
    # Save the final model
    model.save(os.path.join(logs_train_dir, 'final_model.h5'))
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """Plot the training and validation accuracy and loss
    
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

def main():
    # Default paths for Google Colab
    # Users can modify these paths in the notebook
    train_dir = '/content/train_data/'
    logs_train_dir = '/content/model_checkpoints/'
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Run training
    model, history = run_training(train_dir, logs_train_dir)
    
    # Save the model in SavedModel format (TF 2.x preferred format)
    model.save(os.path.join(logs_train_dir, 'saved_model'))
    
    print(f"Training completed. Model saved in {logs_train_dir}")

if __name__ == '__main__':
    main() 