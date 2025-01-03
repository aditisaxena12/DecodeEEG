import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model_simple import build_cnn_model
import numpy as np
from data_loader import load_spectrograms, load_features, train_batch_generator, validation_batch_generator


#load data for one subject
spectrograms = load_spectrograms() # by default subject 1
feature_matrix = load_features() 

# Set parameters
batch_size = 1654
num_samples = feature_matrix.shape[0]  # Total samples in dataset
steps_per_epoch = num_samples * 3 // batch_size  # 3 EEG sets for training
validation_steps = num_samples // batch_size    # 1 EEG set for validation
nb_epoch = 10


# Build the CNN model
model = build_cnn_model(input_shape=(17, 26, 26))

# Train the model
history = model.fit(
    train_batch_generator(spectrograms, feature_matrix, batch_size=batch_size),
    epochs=nb_epoch,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
    validation_data=validation_batch_generator(spectrograms, feature_matrix, batch_size=batch_size),
    validation_steps=validation_steps
)
# Print training and validation accuracy
print("Training Accuracy: {:.2f}%".format(100 * (1 - history.history['loss'][-1])))
print("Validation Accuracy: {:.2f}%".format(100 * (1 - history.history['val_loss'][-1])))

# Save the trained model
model.save("/home/aditis/decodingEEG/DecodeEEG/data/results/cnn_model_simple_trained.h5")

# Plot training vs validation loss (optional)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()