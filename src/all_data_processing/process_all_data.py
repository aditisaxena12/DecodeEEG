import numpy as np
import h5py
import os

path_to_feature_vectors = "/home/aditis/decodingEEG/DecodeEEG/data/feature_vectors/"
path_to_spectrograms = "/home/aditis/decodingEEG/DecodeEEG/data/spectrograms/"

num_samples = 16540 * 4 * 10
num_subjects = 10
num_images_per_sub = 16540 * 4
dim = [1, 26, 26, 17]  # Spectrogram input dimensions
feature_vector_dim = 512  # Ground truth feature vector dimension

subject_ids = []
block_ids = []
specs = []
feats = []

# Load all image vectors
path_to_features = "/home/aditis/decodingEEG/DecodeEEG/data/feature_vectors"

# Initialize an empty list to store the loaded arrays
data = []

classes = os.listdir(path_to_features + "/training/")

for clas in classes:
    feature_path = path_to_features + "/training/" + clas
    files = os.listdir(feature_path)
    for file in files:
        if file.endswith('.npy'):  # Ensure it's a .npy file
            file_path = feature_path + "/" + file
            array = np.load(file_path)  # Load the .npy file
            data.append(array)  # Append to the list

# Convert the list of arrays into a matrix
feature_matrix = np.vstack(data)  # Stack arrays vertically
print(feature_matrix.shape)

# Iterate over subjects and spectrograms
for i in range(num_subjects):
    sub_ind = i + 1
    path_to_spec = "/home/aditis/decodingEEG/DecodeEEG/data/spectrograms/sub-" + "{:02}".format(sub_ind) + "/"
    with h5py.File(path_to_spec + 'spectrograms_train.h5', 'r') as f:
        spectrograms = f['spectrograms']  # Reference to the spectrogram dataset
        print(f"Spectrogram dataset shape: {spectrograms.shape} for subject: {sub_ind}")

        num_images, num_trials, depth, height, width = spectrograms.shape
        for j in range(num_images):
            block_ind = j
            for k in range(num_trials):
                subject_ids.append(sub_ind)
                block_ids.append(block_ind)
                feature_vector = feature_matrix[j, :]
                spectrogram = spectrograms[j, k, :, :, :]
                feats.append(feature_vector)
                specs.append(spectrogram)

# Saving data to HDF5 files with chunking
feats_array = np.vstack(feats)  # Shape: (num_samples, feature_vector_dim)
specs_array = np.array(specs)   # Shape: (num_samples, height, width, depth)
subject_ids_array = np.array(subject_ids)  # Shape: (num_samples,)
block_ids_array = np.array(block_ids)  # Shape: (num_samples,)

# Define file paths for saving
feats_file_path = "/home/aditis/decodingEEG/DecodeEEG/data/processed_data/feats_data.h5"
specs_file_path = "/home/aditis/decodingEEG/DecodeEEG/data/processed_data/specs_data.h5"
subject_ids_file_path = "/home/aditis/decodingEEG/DecodeEEG/data/processed_data/subject_ids_data.h5"
block_ids_file_path = "/home/aditis/decodingEEG/DecodeEEG/data/processed_data/block_ids_data.h5"

# Save feats data to HDF5 with chunking and compression
with h5py.File(feats_file_path, 'w') as f:
    f.create_dataset('features', data=feats_array, chunks=(32, feature_vector_dim))
    print(f"Saved feats data to {feats_file_path}")


# Save subject IDs data to HDF5 with chunking
with h5py.File(subject_ids_file_path, 'w') as f:
    f.create_dataset('subject_ids', data=subject_ids_array, chunks=(32,))
    print(f"Saved subject IDs data to {subject_ids_file_path}")

# Save block IDs data to HDF5 with chunking
with h5py.File(block_ids_file_path, 'w') as f:
    f.create_dataset('block_ids', data=block_ids_array, chunks=(32,))
    print(f"Saved block IDs data to {block_ids_file_path}")


# Save specs data to HDF5 with chunking and compression
with h5py.File(specs_file_path, 'w') as f:
    f.create_dataset('spectrograms', data=specs_array, chunks=(32, depth, height, width))
    print(f"Saved spectrograms data to {specs_file_path}")
