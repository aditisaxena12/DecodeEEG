import numpy as np
import os

def load_eeg(subject_id = "01", training = True):
    '''
    Function to load the EEG data of one subject
    Output : np matrix (16540 x 4 x 17 x 100) for training, (200 x 80 x 17 x 100) for test

    '''
    path_to_eeg = "/home/aditis/decodingEEG/DecodeEEG/data/PreprocessedEEG/sub-"+subject_id
    if training:
        file_path = path_to_eeg + '/preprocessed_eeg_training.npy'
    else:
        file_path = path_to_eeg + '/preprocessed_eeg_test.npy'

    data = np.load(file_path, allow_pickle=True).item()
    eeg_data = data["preprocessed_eeg_data"]
    print("Extracted EEG data of subject: ", subject_id)
    print("Shape of EEG", eeg_data.shape)
    return eeg_data


def load_features(training = True):
    # load all image vectors
    path_to_features = "/home/aditis/decodingEEG/DecodeEEG/data/feature_vectors"

    # Initialize an empty list to store the loaded arrays
    data = []

    if training:
        classes = os.listdir(path_to_features+"/training/")
    else:
        classes = os.listdir(path_to_features+"/test/")

    for clas in classes:
        if training:
            feature_path  = path_to_features + "/training/" + clas
        else:
            feature_path  = path_to_features + "/test/" + clas
        files = os.listdir(feature_path)
        for file in files:
            if file.endswith('.npy'):  # Ensure it's a .npy file
                file_path = feature_path +"/"+ file
                array = np.load(file_path)  # Load the .npy file
                data.append(array)         # Append to the list

        
    # Convert the list of arrays into a matrix
    feature_matrix = np.vstack(data)  # Stack arrays vertically

    # Verify the shape
    print("Shape of feature matrix:",feature_matrix.shape)  # Output: (16540, 512) for training, (200, 512) for test

    return feature_matrix
