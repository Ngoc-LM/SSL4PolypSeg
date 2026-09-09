#!/usr/bin/env python
"""Entrypoint for training/finetuning/testing TriSynNet.

Usage:
    python train.py --mode new     --data_path data/polyp_dataset.npz
    python train.py --mode finetune --resume_ckpt checkpoints/last.ckpt
    python train.py --mode test    --resume_ckpt checkpoints/best.ckpt
"""

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from trisynnet.config import TrainingConfig
from trisynnet.system import PolypDataModule, PolypSSLSystem


def build_argparser():
    parser = argparse.ArgumentParser(description="Train/test TriSynNet")
    parser.add_argument("--mode", choices=["new", "finetune", "test"], default="new")
    parser.add_argument("--data_path", type=str, default=TrainingConfig.data_path)
    parser.add_argument("--checkpoint_dir", type=str, default=TrainingConfig.checkpoint_dir)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--batch_train", type=int, default=TrainingConfig.batch_train)
    parser.add_argument("--labeled_ratio", type=float, default=TrainingConfig.labeled_ratio)
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--no_safpm", action="store_true", help="Disable SAFPM.")
    parser.add_argument("--no_d_biomix", action="store_true", help="Disable D-BioMix.")
    parser.add_argument("--loss_type", choices=["trisynergy", "bce_dice"], default=TrainingConfig.loss_type)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    return parser


def main():
    args = build_argparser().parse_args()
    pl.seed_everything(args.seed, workers=True)

    config = TrainingConfig(
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        max_epochs=args.max_epochs,
        batch_train=args.batch_train,
        labeled_ratio=args.labeled_ratio,
        lr=args.lr,
        seed=args.seed,
        use_safpm=not args.no_safpm,
        use_d_biomix=not args.no_d_biomix,
        loss_type=args.loss_type,
        resume_ckpt=args.resume_ckpt,
    )

    datamodule = PolypDataModule(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="trisynnet-{epoch:03d}-{val/Dice:.4f}",
        monitor="val/Dice",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=20,
    )

    if args.mode == "test":
        if not args.resume_ckpt:
            raise ValueError("--resume_ckpt is required for --mode test")
        system = PolypSSLSystem.load_from_checkpoint(args.resume_ckpt, config=config)
        trainer.test(system, datamodule=datamodule)
        return

    if args.mode == "finetune":
        if not args.resume_ckpt:
            raise ValueError("--resume_ckpt is required for --mode finetune")
        system = PolypSSLSystem.load_from_checkpoint(args.resume_ckpt, config=config)
    else:
        system = PolypSSLSystem(config)

    trainer.fit(system, datamodule=datamodule, ckpt_path=args.resume_ckpt if args.mode == "finetune" else None)


if __name__ == "__main__":
    main()
