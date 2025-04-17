import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers

def create_model(input_shape, n_classes=2):
    """Create a CNN model for face anti-spoofing, based on the original TF1.x model
    
    Args:
        input_shape: tuple of input shape (height, width, channels)
        n_classes: number of output classes
        
    Returns:
        model: Keras model
    """
    # Use Keras functional API to match the original model structure
    inputs = layers.Input(shape=input_shape)
    
    # Block 1: Conv1 -> Pool1
    x = layers.Conv2D(64, (7, 7), padding='same', 
                     kernel_initializer=initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)
    x = layers.Dropout(0.5, name='pool1_dropout')(x)
    
    # Block 2: Conv2 -> Pool2
    x = layers.Conv2D(128, (5, 5), padding='same',
                     kernel_initializer=initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)
    x = layers.Dropout(0.5, name='pool2_dropout')(x)
    
    # Block 3: Conv3_1 -> Conv3_2 -> Pool3
    x = layers.Conv2D(256, (3, 3), padding='same',
                     kernel_initializer=initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv3_1')(x)
    x = layers.Conv2D(256, (3, 3), padding='same',
                     kernel_initializer=initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv3_2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='pool3')(x)
    x = layers.Dropout(0.5, name='pool3_dropout')(x)
    
    # Block 4: Conv4_1 -> Conv4_2 -> Pool4
    x = layers.Conv2D(512, (3, 3), padding='same',
                     kernel_initializer=initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv4_1')(x)
    x = layers.Conv2D(512, (3, 3), padding='same',
                     kernel_initializer=initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv4_2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='pool4')(x)
    x = layers.Dropout(0.5, name='pool4_dropout')(x)
    
    # Block 5: Conv5_1 -> Conv5_2 -> Pool5
    x = layers.Conv2D(512, (3, 3), padding='same',
                     kernel_initializer=initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv5_1')(x)
    x = layers.Conv2D(512, (3, 3), padding='same',
                     kernel_initializer=initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                     bias_initializer=tf.keras.initializers.Constant(0.1),
                     activation='relu', name='conv5_2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='pool5')(x)
    x = layers.Dropout(0.5, name='pool5_dropout')(x)
    
    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, 
                    kernel_regularizer=regularizers.l2(0.005),
                    kernel_initializer=initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    activation='relu', name='local3')(x)
    x = layers.Dropout(0.5, name='local3_dropout')(x)
    
    x = layers.Dense(128, 
                    kernel_regularizer=regularizers.l2(0.005),
                    kernel_initializer=initializers.VarianceScaling(
                         scale=1.0, mode='fan_in', distribution='uniform'),
                    bias_initializer=tf.keras.initializers.Constant(0.1),
                    activation='relu', name='local4')(x)
    x = layers.Dropout(0.5, name='local4_dropout')(x)
    
    # Output layer
    outputs = layers.Dense(n_classes, 
                          kernel_regularizer=regularizers.l2(0.005),
                          kernel_initializer=initializers.VarianceScaling(
                              scale=1.0, mode='fan_in', distribution='uniform'),
                          bias_initializer=tf.keras.initializers.Constant(0.1),
                          name='softmax_linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='face_antispoofing_model')
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