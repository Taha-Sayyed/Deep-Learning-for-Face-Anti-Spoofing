import os
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

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
    model_path = os.path.join(output_dir, f"{model_name}_{timestamp}.h5")
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
    
    # Load the best model for evaluation
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
    return model, test_results

if __name__ == "__main__":
    # Get the TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
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
    model, test_results = train_model(args) 