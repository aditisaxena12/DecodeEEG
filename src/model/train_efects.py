import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model_efects import model_ccn
from data_loader import efects_batch_generator
import h5py
import os

path_to_feature_vectors = "/home/aditis/decodingEEG/DecodeEEG/data/feature_vectors/"
path_to_spectrograms = "/home/aditis/decodingEEG/DecodeEEG/data/spectrograms/"

# Simulated data (replace with actual data)
num_samples = 16540 * 4 * 10
num_subjects = 10
num_images_per_sub = 16540 * 4
dim = [1, 26, 26, 17]  # Spectrogram input dimensions
feature_vector_dim = 512  # Ground truth feature vector dimension

subject_ids = []
block_ids = []
specs = []
feats = []

# load all image vectors
path_to_features = "/home/aditis/decodingEEG/DecodeEEG/data/feature_vectors"

# Initialize an empty list to store the loaded arrays
data = []

classes = os.listdir(path_to_features+"/training/")

for clas in classes:
    feature_path  = path_to_features + "/training/" + clas
    files = os.listdir(feature_path)
    for file in files:
        if file.endswith('.npy'):  # Ensure it's a .npy file
            file_path = feature_path +"/"+ file
            array = np.load(file_path)  # Load the .npy file

            data.append(array)         # Append to the list
# Convert the list of arrays into a matrix
feature_matrix = np.vstack(data)  # Stack arrays vertically
print(feature_matrix.shape)

    
# Convert the list of arrays into a matrix
feature_matrix = np.vstack(data)  # Stack arrays vertically

for i in range(num_subjects):
    sub_ind = i+1
    path_to_spec = "/home/aditis/decodingEEG/DecodeEEG/data/spectrograms/sub-" + "{:02}".format(sub_ind) + "/"
    with h5py.File(path_to_spec+'spectrograms_train.h5', 'r') as f: 
        # Access the dataset
        spectrograms = f['spectrograms']  # This is a reference to the dataset
        print(f"Spectrogram dataset shape: {spectrograms.shape} for subject: {sub_ind}")  # Example: (No of images, No of trials, 17, 26, 26)

        # Reshape the array
        num_images, num_trials, depth, height, width = spectrograms.shape
        for j in range(num_images):
            block_ind = j
            for k in range(num_trials):
                subject_ids.append(sub_ind)
                block_ids.append(block_ind)
                feature_vector = feature_matrix[j,:]
                spectrogram = spectrograms[j, k, :,:,:]
                feats.append(feature_vector)
                specs.append(spectrogram)

print(np.vstack(feats).shape)
print(np.array(specs).shape)               
print(np.array(subject_ids).shape)
print(np.array(block_ids).shape)


# =============== fin ====================

# Model compilation
model = model_ccn(l2_dense=0.001, l2_conv=0.001, dropout=0.2, l2_emb=0.0001, 
                  num_sub=10, num_blocks=16540, dim=[1,26,26,17], emb_rem_index=0, feature_vector_dim=512)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

feature_matrix = np.vstack(feats)
spectrograms = np.array(specs)
subject_ids = np.array(subject_ids)
block_ids = np.array(block_ids)
print("Processed data. Training..")

# Create the batch generator
train_gen = efects_batch_generator(spectrograms, subject_ids, block_ids, feature_matrix, batch_size=32, shuffle=False)

# Train the model
history = model.fit(
    train_gen,
    steps_per_epoch=len(spectrograms) // 32,  # Number of steps per epoch (total samples / batch size)
    epochs=20
)

