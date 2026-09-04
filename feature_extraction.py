# Extracting spectral features from the filtered EEG data
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

# First test on ANT neuro data set
freq_bands = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 12], 'beta': [12, 30]}

def ant_data_test_function():
    ant_data = mne.io.read_raw_ant("dataset/ANT_EEG_flipcup/Ewing_Patrick_2026-08-10_13-07-25_session-01.cnt")
    
    #Crop the data based on the start and end events
    START_CODE = "64"
    END_CODE = "128"
    start_time = ant_data.annotations.onset[ant_data.annotations.description == START_CODE]
    end_time = ant_data.annotations.onset[ant_data.annotations.description == END_CODE]
    ant_data_cropped = ant_data.copy().crop(tmin=start_time[0], tmax=end_time[0])

    frontal_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz']
    frontal_data = ant_data_cropped.copy().pick(frontal_channels)
    spectrum = frontal_data.compute_psd(fmin=4, fmax=8)
    #spectrum.plot(average=False)
    #input("Press Enter to close the plot and exit the script...")

def crop_data(data, start_annotation, stop_annotation, include_stop=True, eps=1e-6):
    ann = data.annotations
    onsets = np.array(ann.onset)
    durations = np.array(ann.duration)
    descs = np.array(ann.description, dtype='<U200')

    starts = onsets[descs == start_annotation]
    stops  = onsets[descs == stop_annotation]

    s = float(starts[0])
    e = float(stops[0]) + (eps if include_stop else -eps)

    # Crop a copy so that og is not modified 
    cropped_data = data.copy().crop(tmin=s, tmax=e)

    # Keep only annotations that lie inside [s, e) and shift their onsets to be relative to new_raw
    inside_mask = (onsets >= s) & (onsets < e)
    new_onsets = (onsets[inside_mask] - s).tolist()
    new_durations = durations[inside_mask].tolist()
    new_descs = descs[inside_mask].tolist()

    new_annotations = mne.Annotations(onset=new_onsets,
                                      duration=new_durations,
                                      description=new_descs)
    cropped_data.set_annotations(new_annotations)
    return cropped_data

def n_back_data_test():
    # Load with the BrainVision format and scale the data to microvolts (V)
    # since data was in microvolts
    n_back_data = mne.io.read_raw_brainvision(r"n_back_dataset\sub-001\eeg\sub-001_task-nback_eeg.vhdr",
    preload=True)
    n_back_data.rescale({'eeg': 1e-6})

    #Preprocess data
    n_back_data.notch_filter(np.arange(50, n_back_data.info['sfreq'] / 2, 50), picks='eeg')
    n_back_data.filter(l_freq=0.5, h_freq=40, picks='eeg')

    # Load the events and attach them as annotations to the raw data
    events_df = pd.read_csv(r"n_back_dataset\sub-001\eeg\sub-001_task-nback_events.tsv", sep='\t')
    onsets = events_df['onset'].astype(float).values                    # seconds relative to recording onset
    durations = events_df['duration'].fillna(0).astype(float).values if 'duration' in events_df.columns else np.zeros(len(events_df))
    if 'trial_type' in events_df.columns:
        descriptions = events_df['trial_type'].astype(str).values
    elif 'value' in events_df.columns:
        descriptions = events_df['value'].astype(str).values
    else:
        descriptions = events_df.index.astype(str).values    # fallback
    ann = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    n_back_data.set_annotations(ann)

    #Crop the data based on the start and end events
    n_back_data_cropped = crop_data(n_back_data, start_annotation= 'started_n_back', stop_annotation= 'finished_n_back')
    
    n_back_data_cropped.plot()
    input("Press Enter to close the plot and exit the script...")

n_back_data_test()