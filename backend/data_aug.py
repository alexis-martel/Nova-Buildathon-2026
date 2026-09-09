def aug_sample(samp, n_augs: int):
    """Runs `n_augs` augmentation methods, randomly chosen from the below list. Returns the augmented `samp`.
    Augmentation methods:
        - Gaussian noise
        - Random scaling
        - Time masking
    """
    return

def apply_gaussian_noise(samp, noise_std=0.05, per_channel=True, seed=42):
    return

def apply_amplitude_scaling(samp, scale_by=(0.9, 1.1), per_channel=True, seed=42):
    return

def apply_time_masking(samp, max_mask_frac=0.1, n_masks=1, fill=0.0, seed=42):
    return

