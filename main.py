from frontend import nback
from frontend import nback_ui
from pathlib import Path

ITEM_DIR = Path("frontend/assets")
NBACK_ITEMS = list(ITEM_DIR.iterdir())


def run_nback(
    n: int,
    assets: list[Path],
    length: int,
    interval: float,
    num_items: int | None = None,
    percent_nback: float = 27,
) -> float:
    """
    Args:
        n:             The `n` of the n-back test
        assets:        List of `pathlib.Path`s pointing to image files
        length:        Number of items in generated n-back sequence
        interval:      Time (in seconds) between which each item is displayed on-screen
        num_items:     Number of items in assets to use in the test (i.e. the number of different items)
        percent_nback: Percentage of items in the n-back sequence that will be positive hits
    Returns: accuracy
    """
    seq = nback.generate_n_back_seq(n, assets, length, num_items, percent_nback)
    pat_picks = nback_ui.start_n_back_ui(seq, interval, f"N-Back Test (N = {n})")
    pos_picks = nback.get_positive_n_back_picks(n, seq)
    score = nback.compute_score(pat_picks, pos_picks, length)
    print(f"Patient accuracy is: {score}")
