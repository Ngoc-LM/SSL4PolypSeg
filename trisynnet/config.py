"""Training configuration for TriSynNet."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    # Data
    data_path: str = "data/polyp_dataset.npz"
    checkpoint_dir: str = "checkpoints"
    img_size: int = 256
    val_size: int = 352
    batch_train: int = 16
    batch_val: int = 8
    num_workers: int = 4
    labeled_ratio: float = 0.1

    # Which split to monitor for checkpointing / early stopping. Must be the
    # in-domain held-in validation split ("val"); pointing this at one of the
    # out-of-domain test_* splits would leak test performance into model
    # selection.
    val_dataset: str = "val"

    # Model components
    use_safpm: bool = True
    use_d_biomix: bool = True
    loss_type: str = "trisynergy"  # "trisynergy" or "bce_dice"

    # SAFPM
    bank_size: int = 30
    feature_dim: int = 512
    num_sectors: int = 8
    safpm_base_gamma: float = 0.05
    safpm_max_gamma: float = 0.20
    safpm_temperature: float = 0.07
    pyramid_levels: int = 3

    # D-BioMix
    mix_prob: float = 0.5
    grid_size: int = 6
    deform_magnitude: float = 0.15

    # Mean Teacher
    ema_alpha: float = 0.99
    rampup_epochs: int = 20
    tau: float = 0.85

    # Optimization
    max_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cuda"

    # Resume / finetune
    resume_ckpt: Optional[str] = None


def guard_val_dataset(config: TrainingConfig) -> None:
    """Raises if `val_dataset` is pointed at an out-of-domain test split.

    Model selection (checkpointing, early stopping, LR scheduling) must only
    ever see the held-in validation split. Evaluating on a test_* split is
    fine at test time, but using it here would leak test information into
    training decisions.
    """
    if config.val_dataset != "val":
        raise ValueError(
            f"config.val_dataset={config.val_dataset!r} is not the held-in "
            "validation split. Model selection must be driven by 'val'; "
            "evaluate on test_* splits only via the separate test step."
        )
