"""D-BioMix: Deformable Bio-Harmonized Mixing.

Mixes a labeled (source) image/mask pair into an unlabeled (target) image
using a smooth control-point deformation field, LAB-space luminance
matching, and SAFPM-based frequency harmonization, then refines the mixed
pseudo-label with a second teacher pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class D_BioMix(nn.Module):
    """
    Args:
        safpm_module: a SAFPM instance, used to harmonize the mixed image's
            high-frequency content after mixing.
        mix_prob: base probability of applying a mix on a given step
            (ramped up over training via a curriculum schedule).
        grid_size: resolution of the sparse control-point grid used to
            generate the deformation field.
        tau: confidence threshold for pseudo-labels, shared with the
            training loop's teacher pseudo-labeling threshold.
        deform_magnitude: base magnitude of the control-point deformation.

    forward() expects two index-aligned pairs:
        target_images/target_masks: the unlabeled batch (target of mixing)
            and its current teacher pseudo-label.
        source_images/source_masks: the labeled batch (source of the mix)
            and its ground-truth mask, at the same batch positions.

    The final mixed label is the union of three components: the original
    target pseudo-label, the warped ground-truth mask from the source
    image, and a second teacher pass on the mixed+harmonized image.
    """

    def __init__(self, safpm_module, mix_prob=0.5, grid_size=6, tau=0.85, deform_magnitude=0.15):
        super().__init__()
        self.safpm = safpm_module
        self.target_mix_prob = mix_prob
        self.grid_size = grid_size
        self.tau = tau
        self.deform_magnitude = deform_magnitude

        self.register_buffer("rgb2xyz", torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ]).view(3, 3, 1, 1).float())

        self.register_buffer("xyz2rgb", torch.tensor([
            [3.240479, -1.537150, -0.498535],
            [-0.969256, 1.875992, 0.041556],
            [0.055648, -0.204043, 1.057311],
        ]).view(3, 3, 1, 1).float())

    def _rgb_to_lab(self, x):
        xyz = F.conv2d(x, self.rgb2xyz)
        xyz = xyz / torch.tensor([0.95047, 1.0, 1.08883], device=x.device).view(1, 3, 1, 1)

        mask = xyz > 0.008856
        xyz = torch.where(mask, torch.pow(xyz, 1 / 3), 7.787 * xyz + 16 / 116)

        L = 116 * xyz[:, 1:2, :, :] - 16
        a = 500 * (xyz[:, 0:1, :, :] - xyz[:, 1:2, :, :])
        b = 200 * (xyz[:, 1:2, :, :] - xyz[:, 2:3, :, :])
        return torch.cat([L, a, b], dim=1)

    def _lab_to_rgb(self, x):
        L, a, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = (L + 16) / 116
        x_ = a / 500 + y
        z = y - b / 200

        xyz = torch.cat([x_, y, z], dim=1)
        mask = xyz > 0.2068966
        xyz = torch.where(mask, torch.pow(xyz, 3), (xyz - 16 / 116) / 7.787)
        xyz = xyz * torch.tensor([0.95047, 1.0, 1.08883], device=x.device).view(1, 3, 1, 1)

        rgb = F.conv2d(xyz, self.xyz2rgb)
        return torch.clamp(rgb, 0, 1)

    def _match_luminance(self, source_rgb, target_rgb):
        """Transfers L-channel statistics from target to source."""
        src_lab = self._rgb_to_lab(source_rgb)
        tgt_lab = self._rgb_to_lab(target_rgb)

        src_L, tgt_L = src_lab[:, 0:1], tgt_lab[:, 0:1]
        src_mean, src_std = src_L.mean(dim=(2, 3), keepdim=True), src_L.std(dim=(2, 3), keepdim=True)
        tgt_mean, tgt_std = tgt_L.mean(dim=(2, 3), keepdim=True), tgt_L.std(dim=(2, 3), keepdim=True)

        new_L = ((src_L - src_mean) / (src_std + 1e-8)) * tgt_std + tgt_mean
        new_lab = torch.cat([new_L, src_lab[:, 1:]], dim=1)
        return self._lab_to_rgb(new_lab)

    def _generate_sparse_deformation(self, B, H, W, device, intensity=1.0):
        """Smooth deformation field from a sparse control-point grid,
        upsampled with bicubic interpolation."""
        gs = self.grid_size
        noise = torch.rand(B, 2, gs, gs, device=device) * 2 - 1
        noise = noise * self.deform_magnitude * intensity

        grid_offset = F.interpolate(noise, size=(H, W), mode="bicubic", align_corners=True)
        grid_offset = grid_offset.permute(0, 2, 3, 1)

        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        base_grid = torch.stack([2 * x / (W - 1) - 1, 2 * y / (H - 1) - 1], dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

        return base_grid + grid_offset

    @torch.no_grad()
    def forward(self, target_images, target_masks, source_images, source_masks,
                teacher_model, features_student,
                reliability_mask=None, current_epoch=0, max_epochs=100):
        B, C, H, W = target_images.shape
        device = target_images.device

        assert source_images.shape[0] == B, (
            f"D-BioMix requires index-aligned source/target batches, got "
            f"source={source_images.shape[0]} vs target={B}."
        )

        # Curriculum: mixing probability and deformation intensity both ramp
        # up over the first 80% of training.
        progress = min(1.0, current_epoch / (max_epochs * 0.8 + 1e-6))
        curr_prob = self.target_mix_prob * (0.5 + 0.5 * progress)

        if torch.rand(1).item() > curr_prob:
            return target_images, target_masks, reliability_mask

        images_denorm = self.safpm.denormalize(target_images).clamp(0, 1)
        source_denorm = self.safpm.denormalize(source_images).clamp(0, 1)

        source_matched = self._match_luminance(source_denorm, images_denorm)

        grid = self._generate_sparse_deformation(B, H, W, device, intensity=0.5 + 0.5 * progress)
        source_warped = F.grid_sample(source_matched, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
        mask_warped = F.grid_sample(source_masks.float(), grid, mode="nearest", padding_mode="reflection", align_corners=True)

        mix_mask = mask_warped
        mixed_images_raw = images_denorm * (1 - mix_mask) + source_warped * mix_mask
        mixed_masks = torch.max(target_masks, mask_warped)

        mixed_images_norm = self.safpm.normalize(mixed_images_raw)
        mixed_images_final = self.safpm.harmonize(mixed_images_norm, features_student)

        logits = teacher_model(mixed_images_final)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = torch.sigmoid(logits)

        pseudo_mask = (pred > self.tau).float()

        # Union of: the original target pseudo-label, the warped source
        # ground-truth, and this second teacher pass on the mixed image.
        final_masks = torch.max(mixed_masks, pseudo_mask)

        mixed_reliability = None
        if reliability_mask is not None:
            mixed_reliability = (1 - mask_warped) * reliability_mask + mask_warped * 1.0

        return mixed_images_final, final_masks, mixed_reliability
