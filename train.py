#!/usr/bin/env python
"""Entry point for training, resuming and testing TriSynNet.

Usage:
    python train.py --mode new    --data_path data/polypData.npz
    python train.py --mode resume --resume_ckpt checkpoints/last.ckpt
    python train.py --mode test   --resume_ckpt checkpoints/<best>.ckpt
"""

import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from trisynnet.config import TrainingConfig
from trisynnet.system import PolypDataModule, PolypSSLSystem


def build_argparser():
    d = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train/test TriSynNet")
    parser.add_argument("--mode", choices=["new", "resume", "test"], default="new")
    parser.add_argument("--data_path", type=str, default=d.data_path)
    parser.add_argument("--checkpoint_dir", type=str, default=d.checkpoint_dir)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=d.max_epochs)
    parser.add_argument("--batch_train", type=int, default=d.batch_train,
                        help="Effective batch size, split equally into labeled and unlabeled samples.")
    parser.add_argument("--labeled_ratio", type=float, default=d.labeled_ratio)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--num_workers", type=int, default=d.num_workers)
    parser.add_argument("--no_safpm", action="store_true", help="Disable SAFPM.")
    parser.add_argument("--no_d_biomix", action="store_true", help="Disable D-BioMix.")
    parser.add_argument("--loss_type", choices=["trisynergy", "bce_dice"], default=d.loss_type)
    parser.add_argument("--edge_gradient", choices=["sobel", "forward_diff"], default=d.edge_gradient,
                        help="Spatial-gradient operator of the Edge Alignment term.")
    parser.add_argument("--gamma_scale", type=float, default=d.gamma_scale,
                        help="gamma_c = 0.05 + 0.15 * tanh(gamma_scale * d_c).")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--limit_test_batches", type=float, default=1.0, help=argparse.SUPPRESS)
    return parser


def main():
    args = build_argparser().parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    config = TrainingConfig(
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        max_epochs=args.max_epochs,
        batch_train=args.batch_train,
        labeled_ratio=args.labeled_ratio,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        use_safpm=not args.no_safpm,
        use_d_biomix=not args.no_d_biomix,
        loss_type=args.loss_type,
        edge_gradient=args.edge_gradient,
        gamma_scale=args.gamma_scale,
        resume_ckpt=args.resume_ckpt,
    )

    datamodule = PolypDataModule(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="trisynnet-{epoch:03d}",
        monitor="val/Dice",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = EarlyStopping(monitor="val/Dice", mode="max", patience=config.early_stop_patience)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=config.precision,
        gradient_clip_val=config.grad_clip_norm,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        log_every_n_steps=10,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )

    if args.mode == "test":
        if not args.resume_ckpt:
            raise ValueError("--resume_ckpt is required for --mode test")
        system = PolypSSLSystem.load_from_checkpoint(args.resume_ckpt, config=config, map_location="cpu")
        trainer.test(system, datamodule=datamodule)
        return

    system = PolypSSLSystem(config)
    if args.mode == "resume":
        if not args.resume_ckpt:
            raise ValueError("--resume_ckpt is required for --mode resume")
        # Restores weights, teacher, SAFPM bank, optimizer, scheduler and counters.
        trainer.fit(system, datamodule=datamodule, ckpt_path=args.resume_ckpt)
    else:
        trainer.fit(system, datamodule=datamodule)


if __name__ == "__main__":
    main()
