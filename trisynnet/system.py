"""PyTorch Lightning orchestration: Mean Teacher training loop wiring
together the ResNet34-UNet backbone, SAFPM, D-BioMix, and the TriSynergy loss."""

import copy
import math

import pytorch_lightning as pl
import torch

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

    The student is trained by gradient descent; the teacher is an EMA copy of
    the student. Per iteration (Section 3 of the manuscript):
      1. the teacher predicts the weakly augmented unlabeled batch, giving the
         pseudo-label (threshold tau) and the entropy-based reliability mask;
      2. teacher features of the labeled batch update the SAFPM memory bank;
      3. SAFPM harmonizes the strongly augmented unlabeled view, using the
         student embedding of that view as the retrieval query;
      4. D-BioMix mixes the labeled strong view into the harmonized unlabeled
         view and builds the mixed label and the aligned reliability mask;
      5. the student is optimized on the labeled view (supervised loss) and on
         the mixed sample (consistency loss weighted by lambda(t)).
    No gradient flows through the bank, the retrieval, D-BioMix, or the teacher.
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

        def make_safpm():
            return SAFPM(
                bank_size=config.bank_size,
                feature_dim=config.feature_dim,
                num_sectors=config.num_sectors,
                base_gamma=config.safpm_base_gamma,
                max_gamma=config.safpm_max_gamma,
                temperature=config.safpm_temperature,
                pyramid_levels=config.pyramid_levels,
                bank_ema_alpha=config.bank_ema_alpha,
                gamma_scale=config.gamma_scale,
            )

        self.safpm = make_safpm() if config.use_safpm else None

        if config.use_d_biomix:
            # Without SAFPM, D-BioMix keeps its own SAFPM instance only for
            # (de)normalization; its bank is never filled, so harmonization
            # inside D-BioMix leaves the mixed image unchanged.
            self.d_biomix = D_BioMix(
                safpm_module=self.safpm if self.safpm is not None else make_safpm(),
                mix_prob=config.mix_prob,
                grid_size=config.grid_size,
                tau=config.tau,
                deform_magnitude=config.deform_magnitude,
            )
        else:
            self.d_biomix = None

        if config.loss_type == "trisynergy":
            self.loss_fn = TriSynergyLoss(max_epochs=config.max_epochs, edge_gradient=config.edge_gradient)
        elif config.loss_type == "bce_dice":
            self.loss_fn = BceDiceLoss()
        else:
            raise ValueError(f"Unknown loss_type: {config.loss_type!r}")

        self.val_dice = []
        self.test_step_outputs = []

    # -- Mean Teacher ---------------------------------------------------------
    def on_train_start(self):
        # The teacher starts as a copy of the student (not on resume).
        if self.global_step == 0:
            self.teacher.load_state_dict(self.student.state_dict())

    @torch.no_grad()
    def _update_teacher(self):
        alpha = self.config.ema_alpha
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_teacher()

    # SAFPM.bank_full is a plain attribute (not a buffer), so it is stored in
    # the checkpoint explicitly; otherwise a resumed run would refill the bank.
    def _safpm_modules(self):
        modules = {}
        if self.safpm is not None:
            modules["safpm"] = self.safpm
        if self.d_biomix is not None and self.d_biomix.safpm is not self.safpm:
            modules["d_biomix_safpm"] = self.d_biomix.safpm
        return modules

    def on_save_checkpoint(self, checkpoint):
        checkpoint["safpm_bank_full"] = {k: bool(m.bank_full) for k, m in self._safpm_modules().items()}

    def on_load_checkpoint(self, checkpoint):
        for k, m in self._safpm_modules().items():
            if k in checkpoint.get("safpm_bank_full", {}):
                m.bank_full = checkpoint["safpm_bank_full"][k]

    def _consistency_weight(self):
        """lambda(t): linear ramp from 0 to 1 over the first `rampup_epochs` epochs."""
        if self.current_epoch >= self.config.rampup_epochs:
            return 1.0
        return self.current_epoch / max(1, self.config.rampup_epochs)

    # -- Training ---------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        weak_l, strong_l, mask_l, _ = batch["labeled"]
        weak_u, strong_u, _, _ = batch["unlabeled"]

        with torch.no_grad():
            self.teacher.eval()
            prob_u = torch.sigmoid(self.teacher(weak_u))
            pseudo_mask = (prob_u > self.config.tau).float()
            eps = 1e-6
            entropy = -(prob_u * torch.log(prob_u + eps) + (1 - prob_u) * torch.log(1 - prob_u + eps))
            reliability_mask = 1.0 - torch.clamp(entropy / math.log(2.0), 0, 1)
            feat_l = self.teacher.extract_features(weak_l)

        if self.safpm is not None:
            self.safpm.update_bank(weak_l, mask_l, feat_l)
            with torch.no_grad():
                query_u = self.student.extract_features(strong_u)
                strong_u = self.safpm.harmonize(strong_u, query_u)

        if self.d_biomix is not None:
            with torch.no_grad():
                query_mix = self.student.extract_features(strong_u)
            mix_images, mix_masks, mix_reliability = self.d_biomix(
                target_images=strong_u,
                target_masks=pseudo_mask,
                source_images=strong_l,
                source_masks=mask_l,
                teacher_model=self.teacher,
                features_student=query_mix,
                reliability_mask=reliability_mask,
                current_epoch=self.current_epoch,
                max_epochs=self.config.max_epochs,
            )
        else:
            mix_images, mix_masks, mix_reliability = strong_u, pseudo_mask, reliability_mask

        logits_l = self.student(strong_l)
        loss_sup = self.loss_fn.compute_supervised(logits_l, mask_l)

        logits_mix = self.student(mix_images)
        loss_cons = self.loss_fn.compute_consistency(
            logits_mix, mix_masks, self.current_epoch, reliability_mask=mix_reliability
        )

        consistency_weight = self._consistency_weight()
        loss = loss_sup + consistency_weight * loss_cons

        self.log_dict({
            "train/loss": loss,
            "train/loss_sup": loss_sup,
            "train/loss_cons": loss_cons,
            "train/consistency_weight": consistency_weight,
        }, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # -- Validation (held-in "val" split) ---------------------------------------
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        probs = torch.sigmoid(self.student(images))
        self.val_dice.append(compute_per_sample_metrics(probs, masks)["Dice"].detach())

    def on_validation_epoch_end(self):
        if not self.val_dice:
            return
        # Mean over all validation images (not over batches).
        self.log("val/Dice", torch.cat(self.val_dice).mean(), prog_bar=True)
        self.val_dice.clear()

    # -- Test ---------------------------------------------------------------------
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images, masks = batch
        probs = torch.sigmoid(self.student(images))
        region_metrics = compute_per_sample_metrics(probs, masks)
        boundary_metrics = compute_boundary_metrics_batch(probs, masks)
        self.test_step_outputs.append({
            "dataloader_idx": dataloader_idx,
            "region": {k: v.detach().cpu() for k, v in region_metrics.items()},
            "boundary": boundary_metrics,
        })

    def on_test_epoch_end(self):
        names = getattr(self.trainer.datamodule, "test_names", None) if self.trainer else None
        by_loader = {}
        for out in self.test_step_outputs:
            by_loader.setdefault(out["dataloader_idx"], []).append(out)
        self.test_results = {}
        for idx, outs in sorted(by_loader.items()):
            tag = names[idx] if names else f"loader{idx}"
            row = {"n_images": int(sum(len(o["region"]["Dice"]) for o in outs))}
            for name in ["Dice", "IoU", "Precision", "Recall"]:
                row[name] = float(torch.cat([o["region"][name] for o in outs]).mean())
            boundary_agg = aggregate_boundary_metrics([o["boundary"] for o in outs])
            for name, stats in boundary_agg.items():
                row[name] = stats["mean"]
                row[f"{name}_n_excluded"] = stats["n_excluded_nan"]
            self.test_results[tag] = row

        # Region metrics in %, HD95/ASSD in pixels, Boundary IoU/F1 as fractions.
        header = f"{'Test set':<14}{'n':>5}{'Dice':>8}{'IoU':>8}{'Pre':>8}{'Rec':>8}{'HD95':>8}{'ASSD':>8}{'BIoU':>8}{'BF1':>8}"
        print("\n" + header + "\n" + "-" * len(header))
        for tag, r in self.test_results.items():
            print(f"{tag:<14}{r['n_images']:>5}"
                  f"{100 * r['Dice']:>8.2f}{100 * r['IoU']:>8.2f}{100 * r['Precision']:>8.2f}{100 * r['Recall']:>8.2f}"
                  f"{r['HD95']:>8.2f}{r['ASSD']:>8.2f}{r['BoundaryIoU']:>8.3f}{r['BoundaryF1']:>8.3f}"
                  + (f"  (HD95/ASSD undefined for {r['HD95_n_excluded']} image(s), excluded)" if r["HD95_n_excluded"] else ""))
        self.test_step_outputs.clear()

    # -- Optimization ---------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            eps=self.config.adamw_eps,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
            min_lr=self.config.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/Dice", "interval": "epoch", "frequency": 1},
        }


class PolypDataModule(pl.LightningDataModule):
    """Wraps `build_semisup_loaders` for use with `pl.Trainer`."""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.train_loaders = None
        self.val_loader = None
        self.test_loaders = None
        self.test_names = None

    def setup(self, stage=None):
        if self.train_loaders is not None:
            return
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
        self.test_names = list(self.test_loaders.keys())

    def train_dataloader(self):
        from pytorch_lightning.utilities import CombinedLoader
        # One epoch is a full pass over the (larger) unlabeled pool, with the
        # labeled stream cycled to match it.
        return CombinedLoader(self.train_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return [self.test_loaders[name] for name in self.test_names]
