"""PyTorch Lightning orchestration: Mean Teacher training loop wiring
together the ResNet34-UNet backbone, SAFPM, D-BioMix, and TriSynergy loss."""

import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .config import TrainingConfig, guard_val_dataset
from .data import build_semisup_loaders
from .losses import BceDiceLoss, TriSynergyLoss
from .metrics import (
    aggregate_boundary_metrics,
    compute_boundary_metrics_batch,
    compute_per_sample_metrics,
)
from .models import SAFPM, D_BioMix, ResNet34UNet_SSL


class PolypSSLSystem(pl.LightningModule):
    """Mean Teacher semi-supervised segmentation system.

    The student is trained by gradient descent; the teacher is an EMA copy
    of the student, used to produce pseudo-labels for the unlabeled batch
    and, when D-BioMix is enabled, to refine pseudo-labels on mixed images.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters(vars(config))
        self.config = config
        guard_val_dataset(config)

        self.student = ResNet34UNet_SSL()
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.safpm = SAFPM(
            bank_size=config.bank_size,
            feature_dim=config.feature_dim,
            num_sectors=config.num_sectors,
            base_gamma=config.safpm_base_gamma,
            max_gamma=config.safpm_max_gamma,
            temperature=config.safpm_temperature,
            pyramid_levels=config.pyramid_levels,
            bank_ema_alpha=config.bank_ema_alpha,
        ) if config.use_safpm else None

        self.d_biomix = D_BioMix(
            safpm_module=self.safpm,
            mix_prob=config.mix_prob,
            grid_size=config.grid_size,
            tau=config.tau,
            deform_magnitude=config.deform_magnitude,
        ) if (config.use_d_biomix and config.use_safpm) else None

        self.loss_fn = TriSynergyLoss(max_epochs=config.max_epochs) if config.loss_type == "trisynergy" else BceDiceLoss()

        self.test_step_outputs = []

    # -- EMA teacher update --------------------------------------------------
    @torch.no_grad()
    def _update_teacher(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.config.ema_alpha)
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)
        for t_buf, s_buf in zip(self.teacher.buffers(), self.student.buffers()):
            t_buf.data.copy_(s_buf.data)

    def _rampup_weight(self):
        if self.current_epoch >= self.config.rampup_epochs:
            return 1.0
        progress = self.current_epoch / max(1, self.config.rampup_epochs)
        return float(torch.exp(torch.tensor(-5.0 * (1.0 - progress) ** 2)))

    # -- Training --------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        weak_l, strong_l, mask_l, _ = batch["labeled"]
        weak_u, strong_u, _, _ = batch["unlabeled"]

        logits_l = self.student(strong_l)
        loss_sup = self.loss_fn.compute_supervised(logits_l, mask_l)

        with torch.no_grad():
            teacher_logits_u = self.teacher(weak_u)
            teacher_prob_u = torch.sigmoid(teacher_logits_u)
            pseudo_mask = (teacher_prob_u > self.config.tau).float()
            reliability_mask = (
                (teacher_prob_u > self.config.tau) | (teacher_prob_u < 1 - self.config.tau)
            ).float()

        mix_images, mix_masks, mix_reliability = strong_u, pseudo_mask, reliability_mask
        if self.safpm is not None:
            student_features_l = self.student.extract_features(weak_l)
            self.safpm.update_bank(weak_l, mask_l, student_features_l)

        if self.d_biomix is not None:
            student_features_u = self.student.extract_features(weak_u)
            mix_images, mix_masks, mix_reliability = self.d_biomix(
                target_images=weak_u,
                target_masks=pseudo_mask,
                source_images=weak_l,
                source_masks=mask_l,
                teacher_model=self.teacher,
                features_student=student_features_u,
                reliability_mask=reliability_mask,
                current_epoch=self.current_epoch,
                max_epochs=self.config.max_epochs,
            )
        elif self.safpm is not None:
            student_features_u = self.student.extract_features(weak_u)
            mix_images = self.safpm.harmonize(strong_u, student_features_u)

        logits_u = self.student(mix_images)
        loss_cons = self.loss_fn.compute_consistency(
            logits_u, mix_masks, self.current_epoch, reliability_mask=mix_reliability
        )

        consistency_weight = self._rampup_weight()
        loss = loss_sup + consistency_weight * loss_cons

        self.log_dict({
            "train/loss": loss,
            "train/loss_sup": loss_sup,
            "train/loss_cons": loss_cons,
            "train/consistency_weight": consistency_weight,
        }, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_teacher()

    # -- Validation --------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self.student(images)
        probs = torch.sigmoid(logits)
        metrics = compute_per_sample_metrics(probs, masks)
        for name, values in metrics.items():
            self.log(f"val/{name}", values.mean(), prog_bar=(name == "Dice"), on_epoch=True, sync_dist=True)
        return metrics

    # -- Test --------------------------------------------------------------------
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images, masks = batch
        logits = self.student(images)
        probs = torch.sigmoid(logits)

        region_metrics = compute_per_sample_metrics(probs, masks)
        boundary_metrics = compute_boundary_metrics_batch(probs, masks)

        self.test_step_outputs.append({
            "dataloader_idx": dataloader_idx,
            "region": {k: v.detach().cpu() for k, v in region_metrics.items()},
            "boundary": boundary_metrics,
        })
        return region_metrics

    def on_test_epoch_end(self):
        by_loader = {}
        for out in self.test_step_outputs:
            by_loader.setdefault(out["dataloader_idx"], []).append(out)
        for idx, outs in by_loader.items():
            for name in ["Dice", "IoU", "Precision", "Recall"]:
                vals = torch.cat([o["region"][name] for o in outs])
                self.log(f"test/loader{idx}/{name}", vals.mean())
            boundary_agg = aggregate_boundary_metrics([o["boundary"] for o in outs])
            for name, stats in boundary_agg.items():
                self.log(f"test/loader{idx}/{name}", stats["mean"])
        self.test_step_outputs.clear()

    # -- Optimization --------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, min_lr=self.config.min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/Dice"},
        }


class PolypDataModule(pl.LightningDataModule):
    """Wraps `build_semisup_loaders` for use with `pl.Trainer`."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.train_loaders = None
        self.val_loader = None
        self.test_loaders = None

    def setup(self, stage=None):
        self.train_loaders, self.val_loader, self.test_loaders = build_semisup_loaders(
            data_path=self.config.data_path,
            batch_train=self.config.batch_train,
            batch_val=self.config.batch_val,
            labeled_ratio=self.config.labeled_ratio,
            img_size=self.config.img_size,
            val_size=self.config.val_size,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
        )

    def train_dataloader(self):
        from pytorch_lightning.utilities import CombinedLoader
        # `max_size_cycle`: one epoch is a full pass over the (larger)
        # unlabeled pool, with the labeled stream cycled to match it.
        return CombinedLoader(self.train_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return list(self.test_loaders.values())
