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


def efects_batch_generator(spectrograms, subject_ids, block_ids, feature_matrix, batch_size=32, shuffle=False):
    """
    Batch generator for spectrograms, subject ids, block ids, and corresponding feature vectors.
    Yields batches of data and labels in the required format for the model.
    """
    num_samples = spectrograms.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Extract batches of spectrograms, subject ids, and block ids
        spectrogram_batch = spectrograms[batch_indices]
        subject_batch = subject_ids[batch_indices]
        block_batch = block_ids[batch_indices]
        feature_batch = feature_matrix[batch_indices]
        
        # Reshape spectrogram_batch to the required shape (batch_size, 26, 26, 17)
        spectrogram_batch_reshaped = spectrogram_batch.transpose(0, 3, 1, 2)  # (batch_size, 26, 26, 17)
        
        # Prepare inputs dictionary
        inputs = {
            "input_1": spectrogram_batch_reshaped,  # Shape (batch_size, 26, 26, 17)
            "input_2": subject_batch,                 # Shape (batch_size,)
            "input_3": block_batch                      # Shape (batch_size,)
        }
        
        
        # Yield the inputs and targets as a tuple
        yield inputs, feature_batch