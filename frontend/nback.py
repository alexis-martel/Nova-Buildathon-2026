import random
from pathlib import Path


def generate_n_back_seq(
    n: int,
    assets: list[Path],
    length: int,
    num_items: int | None = None,
    percent_nback: float = 27,
) -> list[Path]:
    """Generate a `n`-Back test of `num_items` in `assets`. If `num_items` is `None`, all items of `assets`  will be used.  The test sequence will be `length` questions long. A positive n-back will be enforced randomly on `percent_nback`% of items."""
    seq = []
    # Clip `assets` to `num_items`
    if num_items:
        assets = assets[:num_items]
    # Generate initial sequence
    for i in range(length):
        seq.append(random.choice(assets))
    # Explicitly ensure a few n-back positives
    percent_nback /= 100
    choose_n = int(len(assets) * percent_nback)
    for i in range(choose_n):
        i = random.randint(0, len(assets) - n - 1)
        assets[i + n] = assets[i]

    return seq


def get_positive_n_back_picks(n: int, seq: list[Path]) -> list[int]:
    """Returns a list of all the valid n-back elements (elements who are equal to the element `n`th element back`) in `seq`."""
    res = []
    for i in range(n, len(seq) - 1):
        if seq[i] == seq[i - n]:
            res.append(i)
    return res


def compute_score(
    patient_picks: list[int], positive_picks: list[int], length: int
) -> float:
    """Returns a score (in %) that relates to the accuracy of the test."""
    mistakes = 0
    # Check for any missing correct answers
    for pos_pick in positive_picks:
        if pos_pick not in patient_picks:
            mistakes += 1
    # Check for any extra incorrect answers
    for pat_pick in patient_picks:
        if pat_pick not in positive_picks:
            mistakes += 1
    return ((length - mistakes) / length) * 100
