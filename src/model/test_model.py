import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from tensorflow.keras.models import load_model
import numpy as np
from data_loader import test_batch_generator, load_features, load_spectrograms
from sklearn.metrics.pairwise import cosine_similarity

#load data for one subject
test_eeg_data = load_spectrograms(training = False) # by default subject 1
test_feature_matrix = load_features(training=False) 


# Load the model
model = load_model("/home/aditis/decodingEEG/DecodeEEG/data/results/cnn_model_simple_trained.h5")

# Generate test spectrograms and evaluate
batch_size = 64
steps = np.ceil(test_feature_matrix.shape[0] / batch_size)


# Evaluate the model
loss, mae = model.evaluate(test_batch_generator(test_eeg_data,test_feature_matrix, batch_size), steps=steps)
print(f"Test Loss: {loss}, Test MAE: {mae}")

predicted_features = []
for batch_data,_ in test_batch_generator(test_eeg_data, test_feature_matrix, batch_size):
    batch_predictions = model.predict(batch_data)
    predicted_features.append(batch_predictions)

predicted_features = np.concatenate(predicted_features, axis=0)
predicted_features = predicted_features.reshape(-1, 80, predicted_features.shape[1], predicted_features.shape[2], predicted_features.shape[3])
predicted_features = np.mean(predicted_features, axis=1)
print(f"Predicted Features Shape: {predicted_features.shape}")
print(f"Test Features Shape: {test_feature_matrix.shape}")

similarity_scores = cosine_similarity(predicted_features.reshape(predicted_features.shape[0], -1), test_feature_matrix.reshape(test_feature_matrix.shape[0], -1))
print(f"Average Cosine Similarity: {np.mean(similarity_scores)}")