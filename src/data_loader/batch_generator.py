from eeg_preprocessing import eeg_to_spectrogram
import numpy as np
import h5py

def train_batch_generator(path_to_spectrograms, feature_matrix, batch_size=1654):
    """
    Generator for training data batches.
    Input : EEG data of one subject (16540 x 4 x 17 x 26 x 26), feature matrix of all images (16540 x 512)
    Output : Batches of spectrograms (1654 x 17 x 401 x 75) and feature vectors (1654 x 512)
    """
    with h5py.File(path_to_spectrograms, 'r') as f: 
        # Access the dataset
        spectrograms = f['spectrograms']  # This is a reference to the dataset
        num_samples = spectrograms.shape[0]

        while True:  # Infinite loop to yield batches
            for i in range(spectrograms.shape[1] - 1):  # Iterate over EEG sets (3 sets)
                spec  = spectrograms[:, i, :, :]
                for k in range(0, num_samples, batch_size):  # Iterate over chunks
                    batch_end = min(k + batch_size, num_samples)  # Ensure no overflow
                    spectro = spec[k:batch_end,:,:]
                    features = feature_matrix[k:batch_end, :]
                    yield (spectro, features)  # Yield the batch



def validation_batch_generator(path_to_spectrograms, feature_matrix, batch_size=1654):
    """
    Generator for validation data batches.
    Input : EEG data of one subject (16540 x 4 x 17 x 26 x 26), feature matrix of all images (16540 x 512)
    Output : Batches of spectrograms (1654 x 17 x 401 x 75) and feature vectors (1654 x 512)
    """
    with h5py.File(path_to_spectrograms, 'r') as f: 
        # Access the dataset
        spectrograms = f['spectrograms']  # This is a reference to the dataset
        num_samples = spectrograms.shape[0]
        spec  = spectrograms[:, 3, :, :]
        while True:  # Infinite loop to yield batches
            for k in range(0, num_samples, batch_size):  # Iterate over chunks
                batch_end = min(k + batch_size, num_samples)  # Ensure no overflow
                spectro = spec[k:batch_end,:,:]
                features = feature_matrix[k:batch_end, :]
                yield (spectro, features)  # Yield the batch


# Define generator for spectrograms
def test_batch_generator(path_to_spec, num_samples, batch_size=20):
    with h5py.File(path_to_spec, 'r') as f: 
        spectrograms = f['spectrograms'] 
        for i in range(0, num_samples, batch_size):
            spectro_batch = spectrograms[i:i+batch_size, :, :, :]
            yield spectro_batch