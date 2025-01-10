import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from decoder import build_decoder_model
import numpy as np
from data_loader import image_batch_generator


path_to_features = "/home/aditis/decodingEEG/DecodeEEG/data/feature_vectors/training"
path_to_images = "/home/aditis/decodingEEG/DecodeEEG/data/images/training_images"

# Build the CNN model
model = build_decoder_model(input_shape=(512,), output_shape=(500, 500, 3))

# Train the model
history = model.fit(
    image_batch_generator(path_to_images, path_to_features),
    epochs=10,
    steps_per_epoch=1654,
    verbose=1,
)

# Save the trained model
model.save("/home/aditis/decodingEEG/DecodeEEG/data/results/decoder_model_trained.h5")

# Plot training vs validation loss (optional)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')
plt.savefig("/home/aditis/decodingEEG/DecodeEEG/data/results/plots/decoder_training.png")