import numpy as np
from tensorflow.keras.optimizers import Adam
from model_latest import model_ccn
import h5py

path_to_feature_vectors = "/home/aditis/decodingEEG/DecodeEEG/data/feature_vectors/"
path_to_spectrograms = "/home/aditis/decodingEEG/DecodeEEG/data/spectrograms/"

# Simulated data (replace with actual data)
num_samples = 16540 * 4 * 10
num_subjects = 10
num_images_per_sub = 16540 * 4
dim = [1, 26, 26, 17]  # Spectrogram input dimensions
feature_vector_dim = 512  # Ground truth feature vector dimension

subject_ids = []
for i in range(num_subjects):
    sub_ind = i+1
    path_to_spec = "/home/aditis/decodingEEG/DecodeEEG/data/spectrograms/sub-" + "{:02}".format(sub_ind) + "/"
    with h5py.File(path_to_spec+'spectrograms_train.h5', 'r') as f: 
        # Access the dataset
        spectrograms = f['spectrograms']  # This is a reference to the dataset
        print(f"Spectrogram dataset shape: {spectrograms.shape}")  # Example: (N, M, 17, 26, 26)


# =============== trial : random input ====================
# Input spectrograms: shape (num_samples, time_points, frequencies, electrodes)
spectrograms = np.random.rand(num_samples, dim[1], dim[2], dim[3])

# Subject IDs: shape (num_samples,)
subject_ids = [1,2,3,4,5,6,7,8,9,10]

# Block IDs: shape (num_samples,)
block_ids = np.random.randint(0, 37, size=num_samples)

# Ground truth feature vectors: shape (num_samples, feature_vector_dim)
ground_truth_features = np.random.rand(num_samples, feature_vector_dim)

# =============== fin ====================

# Create the model
model = model_ccn(feature_vector_dim=feature_vector_dim)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
history = model.fit(
    {"input_1": spectrograms, "input_2": subject_ids, "input_3": block_ids},
    ground_truth_features,
    batch_size=32,
    epochs=20,
    validation_split=0.2
)
