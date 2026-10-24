"""Training configuration for TriSynNet.

Defaults follow the experimental setting described in the manuscript
(Section 4.2) and the equations of Section 3.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    # Data
    data_path: str = "data/polypData.npz"
    checkpoint_dir: str = "checkpoints"
    img_size: int = 256
    val_size: int = 256
    batch_train: int = 64          # effective batch: 32 labeled + 32 unlabeled
    batch_val: int = 8
    num_workers: int = 2
    labeled_ratio: float = 0.1

    # Split used for checkpointing, early stopping and LR scheduling. Must be
    # the held-in validation split ("val"); using a test_* split would leak
    # test performance into model selection.
    val_dataset: str = "val"

    # Model components
    use_safpm: bool = True
    use_d_biomix: bool = True
    loss_type: str = "trisynergy"  # "trisynergy" or "bce_dice"

    # SAFPM
    bank_size: int = 50
    feature_dim: int = 512
    num_sectors: int = 8
    safpm_base_gamma: float = 0.05
    safpm_max_gamma: float = 0.20
    safpm_temperature: float = 0.07
    pyramid_levels: int = 3
    bank_ema_alpha: float = 0.99
    # gamma_c = base + (max - base) * tanh(gamma_scale * d_c); 1.0 as in the manuscript.
    gamma_scale: float = 1.0

    # D-BioMix
    mix_prob: float = 0.5
    grid_size: int = 6
    deform_magnitude: float = 0.15

    # TriSynergy loss: spatial-gradient operator of the Edge Alignment term,
    # "sobel" (3x3 Sobel, as in the manuscript) or "forward_diff".
    edge_gradient: str = "sobel"

    # Mean Teacher
    ema_alpha: float = 0.99
    rampup_epochs: int = 20        # lambda(t) ramps linearly from 0 to 1
    tau: float = 0.85

    # Optimization
    max_epochs: int = 100
    lr: float = 1e-4
    min_lr: float = 1e-7
    lr_patience: int = 7           # ReduceLROnPlateau on held-in validation Dice
    lr_factor: float = 0.5
    weight_decay: float = 1e-4
    adamw_eps: float = 1e-10
    early_stop_patience: int = 20
    grad_clip_norm: float = 1.0
    precision: str = "32"          # full FP32 precision
    seed: int = 42
    device: str = "cuda"

    # Resume / finetune
    resume_ckpt: Optional[str] = None


def guard_val_dataset(config: TrainingConfig) -> None:
    """Raises if `val_dataset` is not the held-in validation split.

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
