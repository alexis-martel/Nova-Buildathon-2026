# Extracting spectral features from the filtered EEG data
import mne
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA, corrmap, create_ecg_epochs
from scipy.integrate import simpson
import seaborn as sns

# Just some constants
freq_bands = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 12], 'beta': [12, 30]}
frontal_channels = ['F3', 'F4', 'F7', 'F8', 'Fz']

def load_nback_data(file_path):
    n_back_data = mne.io.read_raw_brainvision(file_path, preload=True)
    montage = mne.channels.make_standard_montage('standard_1020')
    n_back_data.drop_channels(['D2', 'D3', 'D4', 'D5']) #These are misc
    n_back_data.set_channel_types({'D1': 'ecg'}) # Set D1 as ECG channel
    n_back_data.set_montage(montage, on_missing='ignore', match_case=False) # Set montage for completeness
    n_back_data.rescale(scalings = {'eeg': 1e-6, 'ecg': 1e-6}) # Scale EEG and ECG to SI units (V)
    return n_back_data

def attach_annotations_from_tsv(raw, tsv_file_path):
    #Attach the annotations from the events.tsv to the raw (Needed for cropping and epoching)
    events_df = pd.read_csv(tsv_file_path, sep='\t')
    onsets = events_df['onset'].astype(float).values                    # seconds relative to recording onset
    durations = events_df['duration'].fillna(0).astype(float).values if 'duration' in events_df.columns else np.zeros(len(events_df))
    if 'trial_type' in events_df.columns:
        descriptions = events_df['trial_type'].astype(str).values
    elif 'value' in events_df.columns:
        descriptions = events_df['value'].astype(str).values
    else:
        descriptions = events_df.index.astype(str).values    # fallback
    ann = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw.set_annotations(ann)

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

def create_nback_epochs(raw, condition, drop_desc='dropped_sample', include_stop=True, concat=False):
    # Extract epochs of data between consecutive boundary annotations of a given condition
    # while excluding any segments that overlap with "dropped_samples
    ann = raw.annotations
    onsets = np.asarray(ann.onset, dtype=float)
    durations = np.asarray(ann.duration, dtype=float)
    descs = np.asarray(ann.description, dtype=str)

    # Find indices/onsets of the n-back annotations (exact match)
    mask_nback = (descs == condition)
    nback_onsets = np.sort(onsets[mask_nback])

    # "dropped_samples" (they have durations)
    mask_drop = (descs == drop_desc)
    dropped_onsets = onsets[mask_drop]
    dropped_ends = dropped_onsets + durations[mask_drop]

    segments = []
    boundaries = []

    # need at least two boundary annotations to form one segment
    if len(nback_onsets) < 2:
        return ([] if not concat else None, [])

    # recording end (maximum allowed tmax)
    rec_end = float(raw.first_time + raw.times[-1]) # fixing the decrepency between cropped uncropped timings

    # iterate consecutive pairs
    for start, stop in zip(nback_onsets[:-1], nback_onsets[1:]):
        seg_start = float(start)
        seg_stop = float(stop)

        # Optionally treat include_stop: keep seg_stop as-is; no eps that can exceed rec_end
        # Clamp seg_stop to recording end to avoid crop errors due to rounding
        if seg_stop > rec_end:
            seg_stop = rec_end

        # ensure a positive interval
        if seg_stop <= seg_start:
            continue

        # check overlap with any dropped interval: overlap if dropped_onset < seg_stop and dropped_end > seg_start
        if dropped_onsets.size > 0:
            overlaps = (dropped_onsets < seg_stop) & (dropped_ends > seg_start)
            if np.any(overlaps):
                continue  # skip this candidate segment

        # crop a copy (include_tmax controls whether the final sample is included)
        seg = raw.copy().crop(tmin=seg_start - raw.first_time, tmax=seg_stop - raw.first_time, include_tmax=include_stop)

        # keep only annotations that lie inside [seg_start, seg_stop)
        inside_mask = (onsets >= seg_start) & (onsets < seg_stop)
        if np.any(inside_mask):
            new_onsets = (onsets[inside_mask] - seg_start).tolist()
            new_durations = durations[inside_mask].tolist()
            new_descs = descs[inside_mask].tolist()
            from mne import Annotations
            seg.set_annotations(Annotations(onset=new_onsets, duration=new_durations, description=new_descs))
        else:
            from mne import Annotations
            seg.set_annotations(Annotations(onset=[], duration=[], description=[]))

        segments.append(seg)
        boundaries.append((seg_start, seg_stop))

    if not segments:
        return ([] if not concat else None, [])

    # concatenate segments for visualization (not analysis)
    if concat:
        concatenated = mne.concatenate_raws(segments)
        return concatenated, boundaries

    return segments, boundaries


def create_all_epochs(raw, drop_desc='dropped_sample', include_stop=True, concat=False):
    conditions = ["1-back", "2-back", "3-back", "4-back"] 
    epochs = {}
    boundaries = {}

    for condition in conditions:
        epochs[condition], boundaries[condition] = create_nback_epochs(raw, condition, drop_desc, include_stop, concat)
    return epochs, boundaries

def compute_theta_power_test(n_back_epochs): 
    theta_power = {}
    for condition, epoch_list in n_back_epochs.items():
        if not epoch_list:
            theta_power[condition] = np.nan
            continue
        F3_list = []
        F4_list = []
        F7_list = []
        F8_list = []
        Fz_list = []
        for epoch in epoch_list:
            # Compute the power spectral density (PSD) for each epoch
            spectrum = epoch.compute_psd(fmin=0.5, fmax=30)
            psd_data, freqs = spectrum.get_data(picks= frontal_channels, exclude='bads',
            fmin= freq_bands['theta'][0], fmax=freq_bands['theta'][1], return_freqs=True)

            # Average the PSDs across epochs and channels
            freq_res = freqs[1] - freqs[0]
            absolute_bandpower = simpson(psd_data, dx=freq_res, axis=-1)
            # Integrate the PSD over the theta band to get total power
            F3_list.append(absolute_bandpower[0])
            F4_list.append(absolute_bandpower[1])
            F7_list.append(absolute_bandpower[2])
            F8_list.append(absolute_bandpower[3])
            Fz_list.append(absolute_bandpower[4])

        F3_mean = np.mean(F3_list)
        F4_mean = np.mean(F4_list) 
        F7_mean = np.mean(F7_list)
        F8_mean = np.mean(F8_list)
        Fz_mean = np.mean(Fz_list)
        theta_power[condition] = {"F3": F3_mean, "F4": F4_mean, "F7": F7_mean, "F8": F8_mean, "Fz": Fz_mean}

    return theta_power

def compute_theta_power_improved(n_back_epochs, frontal_channels=frontal_channels, freq_bands=freq_bands):
    #Compute theta power for every epoch and store them
    # as n_back condition -> electrode -> list of powers for every epoch
    theta_power = {}

    for condition, epoch_list in n_back_epochs.items():

        # Create a dictionary to store theta power from every epoch
        # for each frontal electrode
        channel_power = {
            channel: []
            for channel in frontal_channels
        }

        # Handle conditions with no valid epochs
        if not epoch_list:
            theta_power[condition] = channel_power
            continue

        for epoch in epoch_list:

            # Compute PSD
            spectrum = epoch.compute_psd(
                fmin=0.5,
                fmax=30
            )

            # Extract theta-band PSD
            psd_data, freqs = spectrum.get_data(
                picks=frontal_channels,
                exclude="bads",
                fmin=freq_bands["theta"][0],
                fmax=freq_bands["theta"][1],
                return_freqs=True
            )

            # Integrate PSD over frequency to obtain absolute theta power
            absolute_bandpower = simpson(
                psd_data,
                x=freqs,
                axis=-1)

            # Store theta power for each electrode
            for channel, power in zip(
                frontal_channels,
                absolute_bandpower
            ):
                channel_power[channel].append(power)

        # Store all epoch values for each electrode
        theta_power[condition] = channel_power

    return theta_power

def combine_theta_power(theta_power_dicts):
    # Combine theta power dictionaries into a long-form DataFrame.
    # Each row corresponds to one sample, condition, electrode, and epoch.

    rows = []

    # enumerate starts sample numbering at 1
    for sample_number, theta_power in enumerate(theta_power_dicts, start=1):

        for condition, electrodes in theta_power.items():

            for electrode, epoch_values in electrodes.items():

                for epoch_number, theta_power_value in enumerate(epoch_values, start=1):

                    rows.append({
                        "Sample": sample_number,
                        "Condition": condition,
                        "Electrode": electrode,
                        "Epoch": epoch_number,
                        "Theta_Power": theta_power_value
                    })

    df = pd.DataFrame(rows)

    return df

def n_back_data_test():
    n_back_data = load_nback_data(Path("dataset/n_back_dataset/sub-001/eeg/sub-001_task-nback_eeg.vhdr"))
    eeg_picks = mne.pick_types(n_back_data.info, eeg=True, meg=False, stim=False, eog=False) #only pick eeg channels

    #Preprocess data
    #Filter
    n_back_filtered = n_back_data.copy().notch_filter(np.arange(50, n_back_data.info['sfreq'] / 2, 50))
    n_back_filtered.filter(l_freq=0.5, h_freq=40)
    
    # Combine or choose which candidates to exclude (inspect them)
    # Load the events and attach them as annotations to the raw data
    attach_annotations_from_tsv(n_back_filtered, Path("dataset/n_back_dataset/sub-001/eeg/sub-001_task-nback_events.tsv"))

    #Crop the data based on the start and end events
    n_back_data_cropped = crop_data(n_back_filtered, start_annotation= 'started_n_back', stop_annotation= 'finished_n_back')
    
    # Get epoch of data between n-back annotations:
    n_back_epochs, n_back_boundaries = create_all_epochs(n_back_data_cropped, drop_desc='dropped_samples',
    include_stop=True, concat=False)
    theta_power = compute_theta_power_improved(n_back_epochs)
    df =combine_theta_power([theta_power])
    sns.scatterplot(data=df, x='Condition', y='Mean_Theta_Power', hue='Electrode', style='Electrode', s=100)
    plt.xlabel('Condition')
    plt.ylabel('Mean Theta Power')
    plt.title('Theta Power by Condition')
    plt.show()
    input("Press Enter to close the plot and exit the script...")


def n_back_get_all_theta_powers():
    n_back_paths = pd.read_csv(Path("n_back_dataset\n_back_data_paths.csv"))

    theta_power_list = []
    for row in n_back_paths.itertuples(index=False):
        sample = row.sample
        vhdr_path = Path(row.vhdr_path)
        events_path = Path(row.tsv_path)

        n_back_data = load_nback_data(vhdr_path)
        eeg_picks = mne.pick_types(n_back_data.info, eeg=True, meg=False, stim=False, eog=False) #only pick eeg channels

        #Preprocess data
        #Filter
        n_back_filtered = n_back_data.copy().notch_filter(np.arange(50, n_back_data.info['sfreq'] / 2, 50))
        n_back_filtered.filter(l_freq=0.5, h_freq=40)
        
        # Combine or choose which candidates to exclude (inspect them)
        # Load the events and attach them as annotations to the raw data
        attach_annotations_from_tsv(n_back_filtered, events_path)

        #Crop the data based on the start and end events
        n_back_data_cropped = crop_data(n_back_filtered, start_annotation= 'started_n_back', stop_annotation= 'finished_n_back')
        
        # Get epoch of data between n-back annotations:
        n_back_epochs, n_back_boundaries = create_all_epochs(n_back_data_cropped, drop_desc='dropped_samples',
        include_stop=True, concat=False)
        theta_power = compute_theta_power_improved(n_back_epochs)
        theta_power_list.append(theta_power)

    df = combine_theta_power(theta_power_list)
    df.to_csv(r"n_back_theta_power_features.csv", index=False)
