# Extracting spectral features from the filtered EEG data
import mne
mne.viz.set_browser_backend('qt')
# First test on ANT neuro data set

def test_function():
    ant_data = mne.io.read_raw_ant("dataset/ANT_EEG_flipcup/Ewing_Patrick_2026-08-10_13-07-25_session-01.cnt")
    freq_bands = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 12], 'beta': [12, 30]}
    frontal_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz']
    frontal_data = ant_data.copy().pick(frontal_channels)
    spectrum = frontal_data.compute_psd(fmin=4, fmax=8)
    spectrum.plot(average=False)
    input("Press Enter to close the plot and exit the script...")

test_function()