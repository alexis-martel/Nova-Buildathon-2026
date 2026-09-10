"""
eeg_lsl_recorder.py

EEG acquisition helper built on the Lab Streaming Layer (LSL). Designed to be
driven by a Tkinter GUI that does its timing with root.after() in a separate
module: that module just needs to call start_collection() when the
psychological test begins and stop_collection() when it ends.

This talks ONLY to LSL, never to a vendor SDK, so it will work unmodified
once the ANT Neuro amplifier is connected -- ANT Neuro's eego acquisition
software (eego mylab / ASAlab) can export its data as an LSL outlet, and
that outlet is indistinguishable, from this class's point of view, from any
other LSL EEG stream (including a mock one you use for testing today).

All configuration is passed to the constructor -- there are no module-level
globals to set.
"""

import os
import time
import threading
import tempfile
from typing import Optional, List

import numpy as np
import pylsl
import mne


class EEGRecorder:
    """
    Minimal LSL-backed EEG recorder exposing exactly two calls:
    start_collection() and stop_collection(). All LSL plumbing (resolving
    the outlet, pulling chunks, buffering) runs on a background thread so
    neither call blocks the Tk mainloop for the duration of the recording.
    """

    def __init__(
        self,
        stream_name: Optional[str] = None,
        stream_type: str = "EEG",
        resolve_timeout: float = 5.0,
        pull_timeout: float = 0.5,
        max_samples_per_pull: int = 1024,
        scale_to_volts: float = 1e-6,
        save_dir: Optional[str] = None,
        save_fif: bool = True,
    ):
        """
        stream_name: exact LSL 'name' to resolve against. Leave as None to
            resolve by `stream_type` instead, which is more robust while
            you don't yet know exactly what ANT Neuro's software will call
            its outlet.
        stream_type: LSL 'type' field to resolve against when stream_name
            is None. ANT/eego streams use "EEG".
        resolve_timeout: seconds to wait for the outlet to appear.
        pull_timeout: seconds per pull_chunk() call inside the background
            thread. Also used to bound how long stop_collection() blocks
            while the thread winds down (4x this value).
        max_samples_per_pull: cap on samples per pull_chunk() call.
        scale_to_volts: multiplier applied to incoming samples before
            handing them to MNE. ANT Neuro/eego streams EEG in microvolts
            over LSL, but MNE stores EEG data in SI units (volts)
            internally, so this defaults to 1e-6. Set to 1.0 for a device
            that already streams volts.
        save_dir: directory for the .fif backup written on
            stop_collection(). If None, a fresh temp directory is created
            per recording via tempfile.mkdtemp().
        save_fif: whether to write that .fif backup at all.
        """
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.resolve_timeout = resolve_timeout
        self.pull_timeout = pull_timeout
        self.max_samples_per_pull = max_samples_per_pull
        self.scale_to_volts = scale_to_volts
        self.save_dir = save_dir
        self.save_fif = save_fif

        self._inlet: Optional[pylsl.StreamInlet] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._chunks: List[np.ndarray] = []        # each: (n_samples, n_channels), raw units
        self._timestamps: List[np.ndarray] = []     # LSL clock-corrected timestamps per chunk

        self.sfreq: Optional[float] = None
        self.ch_names: List[str] = []
        self.is_recording: bool = False
        self.last_raw: Optional["mne.io.RawArray"] = None
        self.last_saved_path: Optional[str] = None
        self.first_sample_lsl_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_collection(self) -> None:
        """Resolve the LSL outlet and start pulling samples on a background thread.

        Call once, at the moment the psychological test begins. Returns as
        soon as the stream is found (bounded by resolve_timeout) -- it does
        not block for the duration of the recording, so it's safe to call
        from inside a root.after() callback.
        """
        if self.is_recording:
            raise RuntimeError("start_collection() called while already recording.")

        if self.stream_name:
            streams = pylsl.resolve_byprop("name", self.stream_name, timeout=self.resolve_timeout)
        else:
            streams = pylsl.resolve_byprop("type", self.stream_type, timeout=self.resolve_timeout)

        if not streams:
            raise RuntimeError(
                f"No LSL stream found (name={self.stream_name!r}, type={self.stream_type!r}). "
                "Is the amplifier's LSL exporter (or your mock outlet) running?"
            )

        self._inlet = pylsl.StreamInlet(streams[0], max_buflen=360, recover=True)
        info = self._inlet.info()

        self.sfreq = float(info.nominal_srate())
        self.ch_names = self._extract_channel_names(info)

        self._chunks.clear()
        self._timestamps.clear()
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._pull_loop, daemon=True)
        self._thread.start()
        self.is_recording = True

    def stop_collection(self) -> "mne.io.RawArray":
        """Stop pulling samples and package everything into an MNE Raw object.

        Call once, at the moment the psychological test ends. Blocks briefly
        (bounded by ~4 * pull_timeout) while the background thread winds
        down, then returns an mne.io.RawArray. If save_fif is True, also
        writes a .fif backup and records the path in self.last_saved_path.
        """
        if not self.is_recording:
            raise RuntimeError("stop_collection() called without an active recording.")

        self._stop_event.set()
        self._thread.join(timeout=self.pull_timeout * 4)
        self.is_recording = False

        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:
                pass
            self._inlet = None

        raw = self._build_raw()
        self.last_raw = raw

        if self.save_fif:
            self.last_saved_path = self._save_backup(raw)

        return raw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pull_loop(self) -> None:
        while not self._stop_event.is_set():
            chunk, timestamps = self._inlet.pull_chunk(
                timeout=self.pull_timeout, max_samples=self.max_samples_per_pull
            )
            if timestamps:
                with self._lock:
                    self._chunks.append(np.asarray(chunk, dtype=np.float64))
                    self._timestamps.append(np.asarray(timestamps, dtype=np.float64))

        # One last non-blocking drain so the final fraction of a second
        # sitting in LSL's internal buffer isn't lost.
        chunk, timestamps = self._inlet.pull_chunk(timeout=0.0, max_samples=self.max_samples_per_pull)
        if timestamps:
            with self._lock:
                self._chunks.append(np.asarray(chunk, dtype=np.float64))
                self._timestamps.append(np.asarray(timestamps, dtype=np.float64))

    @staticmethod
    def _extract_channel_names(info: "pylsl.StreamInfo") -> List[str]:
        names = []
        ch = info.desc().child("channels").child("channel")
        for i in range(info.channel_count()):
            label = ch.child_value("label")
            names.append(label if label else f"ch{i + 1}")
            ch = ch.next_sibling()
        # Fall back to generic, MNE-safe names if the device didn't publish
        # usable labels (or published duplicates).
        if not all(names) or len(set(names)) != len(names):
            names = [f"EEG{i + 1:03d}" for i in range(info.channel_count())]
        return names

    def _build_raw(self) -> "mne.io.RawArray":
        with self._lock:
            if not self._chunks:
                raise RuntimeError("No samples were collected -- check the LSL connection.")
            data = np.concatenate(self._chunks, axis=0)            # (n_samples, n_channels)
            timestamps = np.concatenate(self._timestamps, axis=0)  # (n_samples,)

        self.first_sample_lsl_time = float(timestamps[0])

        data = (data * self.scale_to_volts).T  # -> (n_channels, n_samples), volts, MNE layout

        mne_info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, mne_info, verbose=False)
        return raw

    def _save_backup(self, raw: "mne.io.RawArray") -> str:
        save_dir = self.save_dir or tempfile.mkdtemp(prefix="eeg_lsl_")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"recording_{int(time.time())}_raw.fif")
        raw.save(path, overwrite=True, verbose=False)
        return path
