from pathlib import Path
import random

ITEM_DIR = Path("./assets")
NBACK_ITEMS = []

# Generate item list

for item in ITEM_DIR.iterdir():
    NBACK_ITEMS.append(item)

def generate_n_back_seq(n: int, assets: list[Path], length: int,  num_items: int | None = None, percent_nback: float = 27) -> list[Path]:
    """ Generate a `n`-Back test of `num_items` in `assets`. If `num_items` is `None`, all items of `assets`  will be used.  The test sequence will be `length` questions long. A positive n-back will be enforced randomly on `percent_nback`% of items."""
    seq = []
    # Clip `assets` to `num_items`
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

def compute_scores(patient_picks, positive_picks) -> (float, float):
    """Returns a tuple containing accuracy (% of correct picks) and completeness (% of correct picks identified)"""
    accuracy
    
    return (accuracy, 
