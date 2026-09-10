"""
mock_eeg_outlet.py

Stand-in for the ANT Neuro amplifier while you don't have the hardware yet.
Publishes a plain LSL EEG outlet with the same 'type' (and a realistic
channel layout) that eeg_lsl_recorder.EEGRecorder expects, so you can
develop and test the recorder end-to-end without any device attached.

Typical use: run this in its own terminal/process (or a background thread
of a test script) and point EEGRecorder at it -- to EEGRecorder it looks
identical to a real amplifier's LSL export.
"""

import time
import threading
from typing import Optional, List

import numpy as np
import pylsl


# A standard 32-channel 10-20 layout, just so the recorder's channel-label
# extraction has something realistic to pull from. Sliced/extended
# automatically if you ask for a different channel count.
DEFAULT_32CH = [
    "Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
    "POz", "O1", "O2", "AF7", "AF8",
]


def _resolve_channel_names(n_channels: int, ch_names: Optional[List[str]]) -> List[str]:
    if ch_names is not None:
        if len(ch_names) != n_channels:
            raise ValueError("len(ch_names) must equal n_channels.")
        return list(ch_names)
    if n_channels <= len(DEFAULT_32CH):
        return DEFAULT_32CH[:n_channels]
    return DEFAULT_32CH + [f"EEG{i + 1:03d}" for i in range(len(DEFAULT_32CH), n_channels)]


def run_mock_eeg_outlet(
    sfreq: float = 512.0,
    n_channels: int = 32,
    ch_names: Optional[List[str]] = None,
    amplitude_uv: float = 50.0,
    stream_name: str = "MockEEG",
    stream_type: str = "EEG",
    chunk_size: int = 32,
    duration: Optional[float] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Push synthetic EEG-like data over LSL until stopped.

    Data is Gaussian noise in microvolts (matching what ANT/eego actually
    streams) with a faint 10 Hz sine riding on channel 0, so there's
    something visually recognizable once you plot the recorded result.

    sfreq, n_channels: shape of the fake stream.
    ch_names: channel labels to publish; auto-generated from a standard
        32-channel 10-20 layout if omitted.
    amplitude_uv: standard deviation of the noise, in microvolts.
    stream_name, stream_type: LSL identifiers -- match these against
        EEGRecorder(stream_name=..., stream_type=...).
    chunk_size: samples pushed per loop iteration.
    duration: seconds to run for. None runs until `stop_event` is set (or
        forever / until Ctrl+C when run as a script with no stop_event).
    stop_event: pass a threading.Event to stop this from another thread --
        handy for running the mock outlet inside the same test process as
        EEGRecorder.
    """
    names = _resolve_channel_names(n_channels, ch_names)

    info = pylsl.StreamInfo(
        name=stream_name,
        type=stream_type,
        channel_count=n_channels,
        nominal_srate=sfreq,
        channel_format="float32",
        source_id="mock-eeg-001",
    )
    channels = info.desc().append_child("channels")
    for label in names:
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")

    outlet = pylsl.StreamOutlet(info, chunk_size=chunk_size)

    print(
        f"[mock_eeg_outlet] streaming '{stream_name}' ({stream_type}), "
        f"{n_channels} ch @ {sfreq} Hz -- Ctrl+C to stop"
    )

    t0 = time.time()
    n_pushed = 0
    period = chunk_size / sfreq

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if duration is not None and (time.time() - t0) >= duration:
                break

            t = (n_pushed + np.arange(chunk_size)) / sfreq
            chunk = np.random.normal(0.0, amplitude_uv, size=(chunk_size, n_channels))
            chunk[:, 0] += amplitude_uv * 0.5 * np.sin(2 * np.pi * 10.0 * t)  # 10 Hz "signal" on ch 0

            outlet.push_chunk(chunk.astype(np.float32).tolist())
            n_pushed += chunk_size

            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        print(
            f"[mock_eeg_outlet] stopped after {n_pushed} samples "
            f"({n_pushed / sfreq:.1f}s of data)"
        )


if __name__ == "__main__":
    # Run standalone: `python mock_eeg_outlet.py` -- leave this running in
    # its own terminal while you exercise EEGRecorder from your app.
    run_mock_eeg_outlet()
