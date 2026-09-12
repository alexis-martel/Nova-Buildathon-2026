from pathlib import Path

from backend import feature_extraction
from backend.lsl_recorder import EEGRecorder
from frontend import nback, nback_ui, welcome_ui, graph_ui

ITEM_DIR = Path("frontend/assets")
NBACK_ITEMS = list(ITEM_DIR.iterdir())

def bold(string):
    return f"\033[1m{string}\033[0m"


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
    pos_picks = nback.get_positive_n_back_picks(n, seq)
    accuracy_score = nback.compute_score(pat_picks, pos_picks, length)
    return (accuracy_score, eeg_data)

def compute(data, baseline):
    rel_theta_power = feature_extraction.get_relative_theta_power(data, baseline, frontal_channels =["Fz"], freq_bands={'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 12], 'beta': [12, 30]})
    feature_extraction.predict_cw(rel_theta_power)
    return True

def test_loop():
    """Runs when the start button is pressed in the welcome UI. Controls when to advance to the next level & when to stop"""
    THRESHOLD_PASS_ACC = 0.6
    scores = []
    running = True
    n = 1
    # Initial test (baseline n=1)
    acc_score, baseline_eeg = run_nback(n, NBACK_ITEMS, 20, .1)
    scores.append({"accuracy": acc_score, "high_effort": None})
    baseline=feature_extraction.get_baseline(baseline_eeg, frontal_channels =["Fz"], freq_bands={'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 12], 'beta': [12, 30]})
    # Game loop
    n = 2
    for i in range(5):
        acc_score, eeg_data = run_nback(n, NBACK_ITEMS, 20, .1)
        high_effort = compute(eeg_data, baseline)
        scores.append({"accuracy": acc_score, "high_effort": high_effort})
        print(bold(f"Patient test score: {acc_score}\tHigher effort than baseline: {high_effort}"))
        # Level-change logic
        high_score = acc_score >= THRESHOLD_PASS_ACC 
        if high_score and not high_effort:
            n += 1
            print(bold("Moving to higher level test"))
        elif not high_score and high_effort:
            if n > 1: n -= 1
            else: pass
            print(bold("Moving to lower level test"))
        elif high_score and high_effort:
            n += 1
            print(bold("Moving to higher level test"))
        elif not high_score and not high_effort:
            print(bold("Retrying current level"))
    graph_ui.show_graph_window(scores)

if __name__ == "__main__":
    welcome_ui.show_welcome_ui(test_loop)
