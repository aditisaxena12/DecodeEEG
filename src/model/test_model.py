from tensorflow.keras.models import load_model
import numpy as np
from data_loader import test_batch_generator, load_eeg, load_features
from sklearn.metrics.pairwise import cosine_similarity

#load data for one subject
test_eeg_data = load_eeg(training = False) # by default subject 1
test_feature_matrix = load_features(training=False) 

# Load the model
model = load_model("/path/to/eeg_to_feature_vector_model.h5")

# Generate test spectrograms and evaluate
batch_size = 64
steps = np.ceil(test_eeg_data.shape[0] / batch_size)


# Evaluate the model
loss, mae = model.evaluate(test_batch_generator(test_eeg_data, batch_size), steps=steps)
print(f"Test Loss: {loss}, Test MAE: {mae}")

predicted_features = []
for batch in test_batch_generator(test_eeg_data, batch_size):
    batch_predictions = model.predict(batch)
    predicted_features.append(batch_predictions)

predicted_features = np.concatenate(predicted_features, axis=0)

similarity_scores = cosine_similarity(predicted_features, test_feature_matrix)
print(f"Average Cosine Similarity: {np.mean(similarity_scores)}")