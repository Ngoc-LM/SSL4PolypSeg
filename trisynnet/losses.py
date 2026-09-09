"""TriSynergy loss: supervised BCE+Dice plus a curriculum-weighted blend of
correlation-level consistency (robust to SAFPM style shifts) and edge
contrastive consistency (robust to D-BioMix geometric shifts)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAffinityLoss(nn.Module):
    """Correlation-level consistency: enforces that local pixel
    relationships are preserved even if absolute pixel values change due to
    style transfer (SAFPM harmonization)."""

    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, pred, target):
        p_prob = torch.sigmoid(pred)
        t_prob = target.float()

        p_patches = self.unfold(p_prob)
        t_patches = self.unfold(t_prob)

        center_idx = (self.kernel_size ** 2) // 2
        p_center = p_patches[:, center_idx:center_idx + 1, :]
        t_center = t_patches[:, center_idx:center_idx + 1, :]

        p_affinity = 1.0 - torch.abs(p_patches - p_center)
        t_affinity = 1.0 - torch.abs(t_patches - t_center)

        return F.l1_loss(p_affinity, t_affinity)


class EdgeAlignmentLoss(nn.Module):
    """Edge-aware consistency: aligns predicted boundary orientation with
    the teacher's boundary orientation, weighted by a reliability mask, and
    penalizes spurious gradients away from those boundaries.

    Gradients are computed with fixed 3x3 Sobel kernels (registered as
    non-trainable buffers) rather than simple finite differences, for a
    smoother, noise-robust gradient estimate.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _get_gradients(self, x):
        # x: [B, 1, H, W]. Normalize by the kernel's positive-weight sum (8)
        # so magnitudes stay roughly comparable to a finite-difference scale.
        dx = F.conv2d(x, self.sobel_x, padding=1) / 8.0
        dy = F.conv2d(x, self.sobel_y, padding=1) / 8.0
        return dx, dy

    def forward(self, pred, target, reliability_mask=None):
        pred_prob = torch.sigmoid(pred)
        target = target.float()

        p_dx, p_dy = self._get_gradients(pred_prob)
        t_dx, t_dy = self._get_gradients(target)

        p_mag = torch.sqrt(p_dx ** 2 + p_dy ** 2 + 1e-8)
        t_mag = torch.sqrt(t_dx ** 2 + t_dy ** 2 + 1e-8)

        boundary_mask = (t_mag > 0.1).float()

        dot = p_dx * t_dx + p_dy * t_dy
        cosine_sim = dot / (p_mag * t_mag + 1e-8)
        pos_loss = (1.0 - cosine_sim) * boundary_mask

        if reliability_mask is not None:
            pos_loss = pos_loss * reliability_mask

        neg_loss = p_mag * (1.0 - boundary_mask)

        return pos_loss.mean() + 0.5 * neg_loss.mean()


class TriSynergyLoss(nn.Module):
    def __init__(self, max_epochs=100):
        super().__init__()
        self.max_epochs = max_epochs
        self.bce = nn.BCEWithLogitsLoss()
        self.affinity_loss = LocalAffinityLoss(kernel_size=5)
        self.edge_loss = EdgeAlignmentLoss()
        self.smooth = 1.0

    def _dice_loss(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(pred_prob.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def compute_supervised(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_dice = self._dice_loss(logits, targets)
        return 0.5 * loss_bce + 0.5 * loss_dice

    def compute_consistency(self, logits, targets, current_epoch, reliability_mask=None):
        loss_base_bce = self.bce(logits, targets)
        loss_base_dice = self._dice_loss(logits, targets)
        loss_base = 0.5 * loss_base_bce + 0.5 * loss_base_dice

        loss_corr = self.affinity_loss(logits, targets)
        loss_edge = self.edge_loss(logits, targets, reliability_mask)

        # Curriculum: emphasize global-structure consistency early, fine
        # boundary consistency later.
        progress = current_epoch / (self.max_epochs + 1e-6)
        w_corr = 1.0 - 0.5 * progress
        w_edge = 0.1 + 0.9 * progress

        return loss_base + (w_corr * loss_corr) + (w_edge * loss_edge)


class BceDiceLoss(nn.Module):
    """Simple BCE + Dice loss, without the affinity/edge consistency terms
    (useful as a Mean Teacher baseline)."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = 1.0

    def _dice_loss(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(pred_prob.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def compute_supervised(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self._dice_loss(logits, targets)

    def compute_consistency(self, logits, targets, current_epoch=0, reliability_mask=None):
        loss = 0.5 * self.bce(logits, targets) + 0.5 * self._dice_loss(logits, targets)
        if reliability_mask is not None:
            loss = loss * reliability_mask.mean()
        return loss
