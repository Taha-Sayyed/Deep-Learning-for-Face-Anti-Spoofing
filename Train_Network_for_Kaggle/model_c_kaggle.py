import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_model(input_shape=(128, 128, 3), num_classes=2, dropout_rate=0.5):
    """
    Create a CNN model for face anti-spoofing.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled model
    """
    # Create a sequential model
    model = models.Sequential()
    
    # First convolutional block
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate / 2))
    
    # Second convolutional block
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate / 2))
    
    # Third convolutional block
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate / 2))
    
    # Fourth convolutional block
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate / 2))
    
    # Flatten the output and add fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    if num_classes == 2:
        # Binary classification
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        # Multi-class classification
        model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def compile_model(model, learning_rate=0.001, num_classes=2):
    """
    Compile the model with appropriate loss function and metrics.
    
    Args:
        model (tf.keras.Model): Model to compile
        learning_rate (float): Learning rate for optimizer
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: Compiled model
    """
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Define loss function based on number of classes
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def create_ensemble_model(input_shape=(128, 128, 3), num_models=3, num_classes=2, dropout_rate=0.5):
    """
    Create an ensemble of CNN models for face anti-spoofing.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_models (int): Number of models in the ensemble
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled ensemble model
    """
    # Create input layer
    input_layer = layers.Input(shape=input_shape)
    
    # Create multiple model branches
    model_outputs = []
    
    for i in range(num_models):
        # First convolutional block
        x = layers.Conv2D(32, (3, 3), padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        
        # Second convolutional block
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        
        # Third convolutional block
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Output layer for this branch
        if num_classes == 2:
            output = layers.Dense(1, activation='sigmoid', name=f'output_{i}')(x)
        else:
            output = layers.Dense(num_classes, activation='softmax', name=f'output_{i}')(x)
        
        model_outputs.append(output)
    
    # Average the outputs of all models
    if len(model_outputs) > 1:
        ensemble_output = layers.Average()(model_outputs)
    else:
        ensemble_output = model_outputs[0]
    
    # Create and compile the model
    ensemble_model = models.Model(inputs=input_layer, outputs=ensemble_output)
    
    return ensemble_model 