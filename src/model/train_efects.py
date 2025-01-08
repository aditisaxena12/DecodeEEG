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


# Define the file paths where we saved the data
feats_file_path = "/home/aditis/decodingEEG/DecodeEEG/data/processed_data/feats_data.h5"
specs_file_path = "/home/aditis/decodingEEG/DecodeEEG/data/processed_data/specs_data.h5"
subject_ids_file_path = "/home/aditis/decodingEEG/DecodeEEG/data/processed_data/subject_ids_data.h5"
block_ids_file_path = "/home/aditis/decodingEEG/DecodeEEG/data/processed_data/block_ids_data.h5"


# =============== fin ====================

# Model compilation
model = model_ccn(l2_dense=0.001, l2_conv=0.001, dropout=0.2, l2_emb=0.0001, 
                  num_sub=10, num_blocks=16540, dim=[1,26,26,17], emb_rem_index=0, feature_vector_dim=512)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

print("Processed data. Training..")


# Create the batch generator
train_gen = efects_batch_generator(specs_file_path, subject_ids_file_path, block_ids_file_path, feats_file_path, batch_size=32, shuffle=False)

# Train the model
history = model.fit(
    train_gen,
    steps_per_epoch=661600 // 32,  # Number of steps per epoch (total samples / batch size)
    epochs=20
)

# Save the trained model
model.save("/home/aditis/decodingEEG/DecodeEEG/data/results/cnn_model_efects_trained.h5")

# Plot training vs validation loss (optional)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')
plt.savefig("effects.png")
