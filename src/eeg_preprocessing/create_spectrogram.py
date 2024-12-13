import argparse
import os
import numpy as np
import h5py
from eeg_to_spectrogram import eeg_to_spectrogram


def process_eeg_file(input_file, output_file):
    """
    Processes an EEG .npy file into spectrograms and saves the result in HDF5 format.
    """
    # Load the .npy file
    file = np.load(input_file, allow_pickle=True).item()
    data = file['preprocessed_eeg_data']  # Shape: (N, M, 17, 100)
    print(f"Input data shape: {data.shape}")  # e.g., (16540, 4, 17, 100) or (200,80,17,100)

    N, M, num_channels, num_timepoints = data.shape

    # Create an HDF5 file for saving the spectrograms
    with h5py.File(output_file, 'w') as h5f:
        # Create dataset without compression
        dset = h5f.create_dataset(
            "spectrograms",
            shape=(N, M, num_channels, 401, 75),  # Final desired shape
            dtype=np.float32,
        )

        # Process each EEG signal and save incrementally
        for i in range(N):
            print(i)
            spectrogram_images = np.array([
                eeg_to_spectrogram(data[i, j, :, :]) for j in range(M)
            ])  # Shape: (M, 17, 401, 75)
            dset[i] = spectrogram_images  # Write to HDF5 file incrementally

        print(f"Spectrograms saved to {output_file}")
      
                   

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("--sub", help = "subject number 01 to 10")

    # Read arguments from command line
    args = parser.parse_args()

    subject = "sub-"+ args.sub 

    input_folder = "/home/aditis/decodingEEG/DecodeEEG/data/PreprocessedEEG/" + subject
    output_folder = "/home/aditis/decodingEEG/DecodeEEG/data/spectrograms/" + subject
    #create output folder if doesn't exist
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    #empty old files from output folder
    files= os.listdir (output_folder)
    for file in files:
        os.remove(file)

    input_file_test = input_folder + '/preprocessed_eeg_test.npy'
    output_file_test = output_folder + '/spectrograms_test.h5'

    input_file_train = input_folder + '/preprocessed_eeg_training.npy'
    output_file_train = output_folder + '/spectrograms_train.h5'
    process_eeg_file(input_file_train, output_file_train)
    process_eeg_file(input_file_test, output_file_test)

