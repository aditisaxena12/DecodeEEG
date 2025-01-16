from eeg_preprocessing import eeg_to_spectrogram
import numpy as np
import h5py
import os
import cv2

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
                    features = feature_matrix[k:batch_end, :,:,:]
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
                features = feature_matrix[k:batch_end, :,:,:]
                yield (spectro, features)  # Yield the batch


# Define generator for spectrograms
def test_batch_generator(path_to_spec, feature_matrix, batch_size=20):
    num_samples = feature_matrix.shape[0]
    with h5py.File(path_to_spec, 'r') as f: 
        spectrograms = f['spectrograms'] 
        while True:
            for k in range(spectrograms.shape[1] ):  # Iterate over EEG sets (3 sets)
                spec  = spectrograms[:, k, :, :]
                for i in range(0, num_samples, batch_size):
                    batch_end = min(i + batch_size, num_samples)
                    spectro_batch = spec[i:batch_end, :, :, :]
                    features = feature_matrix[i:batch_end, :,:,:]
                    print(spectro_batch.shape)
                    print(features.shape)
                    yield (spectro_batch, features)
            break


def efects_batch_generator(h5_spectrograms_file, h5_subject_ids_file, h5_block_ids_file, h5_feature_matrix_file, batch_size=32, shuffle=False):
    """
    Batch generator for spectrograms, subject ids, block ids, and corresponding feature vectors.
    Yields batches of data and labels in the required format for the model.
    Reads data from HDF5 files to avoid loading everything into memory.
    """
    while True:
        # Open the HDF5 files in read-only mode
        with h5py.File(h5_spectrograms_file, 'r') as f_specs, \
            h5py.File(h5_subject_ids_file, 'r') as f_subs, \
            h5py.File(h5_block_ids_file, 'r') as f_blocks, \
            h5py.File(h5_feature_matrix_file, 'r') as f_feats:
            
            # Access the datasets in the files
            spectrograms = f_specs['spectrograms']  # Shape: (num_samples, height, width, depth)
            subject_ids = f_subs['subject_ids']     # Shape: (num_samples,)
            block_ids = f_blocks['block_ids']       # Shape: (num_samples,)
            feature_matrix = f_feats['features']    # Shape: (num_samples, feature_vector_dim)
            
            num_samples = spectrograms.shape[0]
            indices = np.arange(num_samples)
            
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Extract batches of data
                spectrogram_batch = spectrograms[batch_indices]
                subject_batch = subject_ids[batch_indices]
                block_batch = block_ids[batch_indices]
                feature_batch = feature_matrix[batch_indices]
                
                # Reshape spectrogram_batch to the required shape (batch_size, 26, 26, 17)
                spectrogram_batch_reshaped = spectrogram_batch.transpose(0, 2, 3, 1)  # (batch_size, 26, 26, 17)
                
                # Prepare inputs dictionary
                inputs = {
                    "input_1": spectrogram_batch_reshaped,  # Shape (batch_size, 26, 26, 17)
                    "input_2": subject_batch,              # Shape (batch_size,)
                    "input_3": block_batch                 # Shape (batch_size,)
                }
                
                # Yield the inputs and targets as a tuple
                yield inputs, feature_batch

def image_batch_generator(path_to_images, path_to_features):
    """
    Generator for image data batches.
    Input : Path to directory containing images, feature matrix of all images (16540 x 512)
    Output : Batches of images (10 x 500 x 500 x 3) and feature vectors (10 x 512)
    """
    # List all class files in the directory
    class_folders = os.listdir(path_to_images)
    
    while True:  # Infinite loop to yield batches
        for i, clas in enumerate(class_folders):
            im_class_path = os.path.join(path_to_images, clas)
            feat_class_path = os.path.join(path_to_features, clas)  # Path to feature vectors
            image_files = os.listdir(im_class_path)
            images = np.array([cv2.resize(cv2.imread(os.path.join(im_class_path, file)), (512,512)) for file in image_files])
            features = np.array([np.load(os.path.join(feat_class_path, file.replace('.jpg', '.npy'))) for file in image_files])
            yield (features,images)  # Yield the batch
