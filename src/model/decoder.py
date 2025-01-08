import tensorflow as tf
from tensorflow.keras import layers, models
from keras.utils.vis_utils import plot_model

def build_decoder_model(input_shape=(512,), output_shape=(500, 500, 3)):
    model = models.Sequential()

    # Input layer
    model.add(layers.InputLayer(input_shape=input_shape))

    # Dense layers to map from 512-dimensional feature vector to a suitable shape for convolutional layers
    model.add(layers.Dense(32 * 32 * 256, activation='relu'))
    model.add(layers.Reshape((32, 32, 256)))

    # Upsampling and convolutional layers to reconstruct the image
    model.add(layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))

    model.add(layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))

    model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))

    model.add(layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))

    model.add(layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))

    # Final convolutional layer to get the desired output shape
    model.add(layers.Conv2DTranspose(output_shape[2], (3, 3), activation='sigmoid', padding='same'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    plot_model(model, to_file='/home/aditis/decodingEEG/DecodeEEG/data/results/decoder_plot.png', show_shapes=True, show_layer_names=True) 

    return model

# Example usage
decoder_model = build_decoder_model()
decoder_model.compile()
decoder_model.summary()
