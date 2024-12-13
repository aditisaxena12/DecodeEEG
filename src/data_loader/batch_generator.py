from eeg_preprocessing import eeg_to_spectrogram
import numpy as np

def train_batch_generator(eeg_data, feature_matrix, batch_size=1654):
    """
    Generator for training data batches.
    Input : EEG data of one subject (16540 x 4 x 17 x 100), feature matrix of all images (16540 x 512)
    Output : Batches of spectrograms (1654 x 17 x 401 x 75) and feature vectors (1654 x 512)
    """
    num_samples = eeg_data.shape[0]

    while True:  # Infinite loop to yield batches
        for i in range(eeg_data.shape[1] - 1):  # Iterate over EEG sets (3 sets)
            eeg = eeg_data[:, i, :, :]
            for k in range(0, num_samples, batch_size):  # Iterate over chunks
                spectro = []
                batch_end = min(k + batch_size, num_samples)  # Ensure no overflow
                for j in range(k, batch_end):
                    Sx2 = eeg_to_spectrogram(eeg[j, :, :])  # Spectrogram calculation
                    spectro.append(Sx2)
                
                spectro = np.stack(spectro)
                features = feature_matrix[k:batch_end, :]
                yield (spectro, features)  # Yield the batch


def validation_batch_generator(eeg_data, feature_matrix, batch_size=1654):
    """
    Generator for validation data batches.
    Input : EEG data of one subject (16540 x 4 x 17 x 100), feature matrix of all images (16540 x 512)
    Output : Batches of spectrograms (1654 x 17 x 401 x 75) and feature vectors (1654 x 512)
    """
    num_samples = eeg_data.shape[0]
    eeg = eeg_data[:, 3, :, :]  # Use the 4th set for validation

    while True:  # Infinite loop to yield batches
        for k in range(0, num_samples, batch_size):  # Iterate over chunks
            spectro = []
            batch_end = min(k + batch_size, num_samples)  # Ensure no overflow
            for j in range(k, batch_end):
                Sx2 = eeg_to_spectrogram(eeg[j, :, :])  # Spectrogram calculation
                spectro.append(Sx2)
            
            spectro = np.stack(spectro)
            features = feature_matrix[k:batch_end, :]
            yield (spectro, features)  # Yield the batch