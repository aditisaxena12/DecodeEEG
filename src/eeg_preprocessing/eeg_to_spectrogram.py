# function to convert eeg (17,100) to spectrogram (17, 401, 75)
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

def eeg_to_spectrogram(eeg):
    """
    Converts a single EEG signal (17x100) into its spectrogram.
    """
    g_std = 12  # standard deviation for Gaussian window in samples
    win = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
    T_x, N = 1 / 100, 100  # Sampling rate: 100 Hz, signal length: 100 samples
    SFT = ShortTimeFFT(win, hop=2, fs=1/T_x, mfft=800, scale_to='psd')

    # Calculate Spectrogram
    Sx2 = SFT.spectrogram(eeg)  # shape: (17, ?, ?)
    return Sx2