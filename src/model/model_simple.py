# model architecture
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.utils.vis_utils import plot_model

def build_cnn_model(input_shape=(17, 26, 26)):
    """
    Builds and compiles a CNN model to map spectrograms to feature vectors.
    
    Args:
        input_shape (tuple): Shape of the input data.
        
    Returns:
        model (tf.keras.Model): Compiled CNN model.
    """
    model = models.Sequential()

    # Input layer
    model.add(layers.InputLayer(input_shape=input_shape))

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))  # 32 filters, kernel size 3x3
    model.add(layers.MaxPooling2D((2, 2)))  # Max pooling (2x2)

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))  # 64 filters, kernel size 3x3
    model.add(layers.MaxPooling2D((2, 2)))  # Max pooling (2x2)

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))  # 128 filters, kernel size 3x3
    model.add(layers.MaxPooling2D((2, 2)))  # Max pooling (2x2)

    # Flatten the output of the last convolutional layer
    model.add(layers.Flatten())

    # Dense layers to map to 512-dimensional feature vector
    model.add(layers.Dense(512, activation='relu'))  # Dense layer with 512 neurons
    model.add(layers.Dropout(0.3))  # Dropout for regularization
    model.add(layers.Dense(512, activation='linear'))  # Output layer with 512 neurons (regression task)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    plot_model(model, to_file='/home/aditis/decodingEEG/DecodeEEG/data/results/model_plot.png', show_shapes=True, show_layer_names=True) 

    return model
