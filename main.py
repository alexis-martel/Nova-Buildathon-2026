from pathlib import Path

from backend.lsl_recorder import EEGRecorder
from frontend import nback, nback_ui, welcome_ui

ITEM_DIR = Path("frontend/assets")
NBACK_ITEMS = list(ITEM_DIR.iterdir())


def run_nback(
    n: int,
    assets: list[Path],
    length: int,
    interval: float,
    num_items: int | None = None,
    percent_nback: float = 27,
) -> (float, float):
    """
    Args:
        n:             The `n` of the n-back test
        assets:        List of `pathlib.Path`s pointing to image files
        length:        Number of items in generated n-back sequence
        interval:      Time (in seconds) between which each item is displayed on-screen
        num_items:     Number of items in assets to use in the test (i.e. the number of different items)
        percent_nback: Rough percentage of items in the n-back sequence that will be positive hits
    Returns: (accuracy, effort_score)
    """
    recorder = EEGRecorder(stream_name="MockEEG")
    seq = nback.generate_n_back_seq(n, assets, length, num_items, percent_nback)
    recorder.start_collection()
    pat_picks = nback_ui.start_n_back_ui(seq, interval, f"N-Back Test (N = {n})")
    eeg_data = recorder.stop_collection()
    effort_score = dummy_compute(eeg_data)
    pos_picks = nback.get_positive_n_back_picks(n, seq)
    accuracy_score = nback.compute_score(pat_picks, pos_picks, length)
    return (accuracy_score, effort_score)

def dummy_compute(data):
    print(data)
    return 3.14159265

def test_loop():
    """Runs when the start button is pressed in the welcome UI. Controls when to advance to the next level & when to stop"""
    # Initial test
    scores = run_nback(1, NBACK_ITEMS, 10, 3)
    print(f"acc: {scores[0]}\teff: {scores[1]}")

if __name__ == "__main__":
    welcome_ui.show_welcome_ui(test_loop)
