import argparse
import os
import h5py
import numpy as np
from scipy.signal import spectrogram  

def eeg_to_spectrogram(input_file, output_file):

    # Load the npy file
    file = np.load(input_file,  allow_pickle=True).item()
    data = file['preprocessed_eeg_data']
    print(data.shape)  # Should output (16540, 4, 17, 100)
    # Parameters for spectrogram
    fs = 100  # Sampling frequency (adjust based on your data)
    nperseg = 50  # Length of each segment (adjust based on EEG characteristics)
     # Create an HDF5 file to store the spectrograms
    with h5py.File(output_file, "w") as hf:
        for i in range(data.shape[0]):  # Iterate over 16540
            for j in range(data.shape[1]):  # Iterate over 4
                group = hf.create_group(f"signal_{i}_{j}")
                for k in range(data.shape[2]):  # Iterate over 17
                    signal = data[i, j, k, :]  # Extract the signal
                    frequencies, times, Sxx = spectrogram(signal, fs, nperseg=nperseg)
                    
                    # Store the spectrogram
                    group.create_dataset(f"channel_{k}_frequencies", data=frequencies)
                    group.create_dataset(f"channel_{k}_times", data=times)
                    group.create_dataset(f"channel_{k}_spectrogram", data=Sxx)


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
    output_file_test = output_folder + '/spectrogram_test.h5'

    input_file_train = input_folder + '/preprocessed_eeg_training.npy'
    output_file_train = output_folder + '/spectrogram_training.h5'

    eeg_to_spectrogram(input_file_test, output_file_test)
    eeg_to_spectrogram(input_file_train, output_file_train)
