"""SAFPM: Semantic-Aware Frequency Profile Matching.

Maintains a memory bank of directional frequency profiles paired with
semantic embeddings, learned from labeled images, and uses attention over
that bank to harmonize the high-frequency content of unlabeled images
towards the "in-domain" frequency statistics.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAFPM(nn.Module):
    def __init__(
        self,
        bank_size=50,
        feature_dim=512,
        num_sectors=8,
        base_gamma=0.05,
        max_gamma=0.20,
        temperature=0.07,
        pyramid_levels=3,
        bank_ema_alpha=0.99,
        device="cuda",
    ):
        super().__init__()
        self.bank_size = bank_size
        self.feature_dim = feature_dim
        self.num_sectors = num_sectors
        self.base_gamma = base_gamma
        self.max_gamma = max_gamma
        self.temperature = temperature
        self.pyramid_levels = pyramid_levels
        self.eps = 1e-8

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.register_buffer("prior_bank", None)
        self.register_buffer("semantic_bank", torch.zeros(bank_size, feature_dim))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        self.bank_full = False
        self.bank_ema_alpha = bank_ema_alpha
        self.map_cache = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        bank_key = prefix + "prior_bank"
        if bank_key in state_dict:
            saved_bank = state_dict[bank_key]
            if self.prior_bank is None or self.prior_bank.shape != saved_bank.shape:
                self.register_buffer("prior_bank", saved_bank.clone())
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                       missing_keys, unexpected_keys, error_msgs)

    # -- Laplacian pyramid -------------------------------------------------
    def _build_pyramid(self, x):
        pyramid = []
        current = x
        for _ in range(self.pyramid_levels - 1):
            down = F.interpolate(current, scale_factor=0.5, mode="bilinear", align_corners=False)
            up = F.interpolate(down, size=current.shape[-2:], mode="bilinear", align_corners=False)
            pyramid.append(current - up)
            current = down
        pyramid.append(current)
        return pyramid

    def _reconstruct_pyramid(self, pyramid):
        current = pyramid[-1]
        for i in range(self.pyramid_levels - 2, -1, -1):
            up = F.interpolate(current, size=pyramid[i].shape[-2:], mode="bilinear", align_corners=False)
            current = up + pyramid[i]
        return current

    # -- Directional frequency profile -------------------------------------
    def _get_coordinate_maps(self, H, W, device):
        key = f"{H}_{W}"
        if self.map_cache is not None and self.map_cache.get("key") == key:
            return self.map_cache

        center_h, center_w = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        y_c = (y - center_h).float()
        x_c = (x - center_w).float()

        radius_map = torch.sqrt(y_c.pow(2) + x_c.pow(2)).long()
        angle_map = torch.atan2(y_c, x_c)
        angle_map = (angle_map + math.pi) / (2 * math.pi)
        angle_map = (angle_map * self.num_sectors).long() % self.num_sectors

        self.map_cache = {"radius": radius_map, "angle": angle_map, "key": key}
        return self.map_cache

    def _compute_directional_profile(self, fft_amp):
        device = fft_amp.device
        B, C, H, W = fft_amp.shape
        maps = self._get_coordinate_maps(H, W, device)

        bin_map = maps["radius"] * self.num_sectors + maps["angle"]
        max_bin = bin_map.max().item() + 1

        fft_flat = fft_amp.view(B, C, -1)
        bin_flat = bin_map.view(-1)

        profile = torch.zeros(B, C, max_bin, device=device)
        count = torch.zeros(max_bin, device=device)
        count.index_add_(0, bin_flat, torch.ones_like(bin_flat, dtype=torch.float32))

        idx = bin_flat.view(1, 1, -1).expand(B, C, -1)
        if idx.max() >= max_bin:
            idx = torch.clamp(idx, max=max_bin - 1)

        profile.scatter_add_(2, idx, fft_flat)
        return profile / (count.view(1, 1, -1) + self.eps)

    def denormalize(self, x):
        return x * self.std + self.mean

    def normalize(self, x):
        return (x - self.mean) / self.std

    # -- Memory bank ---------------------------------------------------------
    @torch.no_grad()
    def update_bank(self, img_tensor, mask_tensor, features):
        """Push the frequency profile of a labeled batch (masked to the
        polyp region) plus its semantic embedding into the memory bank."""
        if mask_tensor.sum() < 10:
            return

        img = self.denormalize(img_tensor)
        pyramid = self._build_pyramid(img)
        high_freq = pyramid[0]

        mask = F.interpolate(mask_tensor.float(), size=high_freq.shape[-2:], mode="nearest")
        high_freq_masked = high_freq * mask

        fft = torch.fft.fft2(high_freq_masked, norm="ortho")
        fft_amp = torch.abs(torch.fft.fftshift(fft, dim=(-2, -1)))
        current_profile = self._compute_directional_profile(fft_amp)
        features = F.normalize(features, dim=1)

        batch_size = img_tensor.shape[0]

        if self.prior_bank is None or current_profile.shape[2] != self.prior_bank.shape[2]:
            self.prior_bank = torch.zeros(
                self.bank_size, current_profile.shape[1], current_profile.shape[2],
                device=img.device,
            )
            self.bank_ptr[0] = 0
            self.bank_full = False

        if not self.bank_full:
            ptr = int(self.bank_ptr)
            indices = torch.arange(ptr, ptr + batch_size) % self.bank_size

            if batch_size > self.bank_size:
                indices = indices[: self.bank_size]
                current_profile = current_profile[: self.bank_size]
                features = features[: self.bank_size]
                batch_size = self.bank_size

            self.prior_bank[indices] = current_profile.detach()
            self.semantic_bank[indices] = features.detach()

            self.bank_ptr[0] = (ptr + batch_size) % self.bank_size
            if self.bank_ptr[0] < ptr:
                self.bank_full = True
        else:
            # Nearest-neighbour EMA update: each sample updates the slot it is
            # most similar to (sequential, so within-batch collisions build on
            # each other rather than silently overwriting).
            alpha = self.bank_ema_alpha
            for i in range(batch_size):
                sim = torch.mv(self.semantic_bank, features[i])
                nearest = sim.argmax()

                self.prior_bank[nearest] = (
                    alpha * self.prior_bank[nearest] + (1 - alpha) * current_profile[i].detach()
                )
                self.semantic_bank[nearest] = (
                    alpha * self.semantic_bank[nearest] + (1 - alpha) * features[i].detach()
                )
                self.semantic_bank[nearest] = F.normalize(
                    self.semantic_bank[nearest].unsqueeze(0), dim=1
                ).squeeze(0)

    @torch.no_grad()
    def harmonize(self, img_tensor, features):
        """Attention-retrieve a matching frequency prior from the bank and
        blend it into the high-frequency band of the input image."""
        if self.prior_bank is None or (not self.bank_full and self.bank_ptr < 5):
            return img_tensor

        img_raw = self.denormalize(img_tensor)
        pyramid = self._build_pyramid(img_raw)

        q = F.normalize(features, dim=1)
        k = self.semantic_bank[: self.bank_size if self.bank_full else self.bank_ptr]
        v = self.prior_bank[: self.bank_size if self.bank_full else self.bank_ptr]

        sim = torch.mm(q, k.t())
        attn = F.softmax(sim / self.temperature, dim=1)
        P_prior = torch.einsum("bk,kcl->bcl", attn, v)

        h_freq = pyramid[0]
        fft = torch.fft.fft2(h_freq, norm="ortho")
        fft_amp = torch.abs(fft)
        fft_phase = torch.angle(fft)
        fft_shifted = torch.fft.fftshift(fft_amp, dim=(-2, -1))

        P_u = self._compute_directional_profile(fft_shifted)

        E_u = torch.sum(P_u, dim=-1, keepdim=True)
        P_u_norm = P_u / (E_u + self.eps)
        P_prior_norm = P_prior / (torch.sum(P_prior, dim=-1, keepdim=True) + self.eps)

        eff_len = min(P_u_norm.shape[-1], P_prior_norm.shape[-1])
        P_u_norm = P_u_norm[..., :eff_len]
        P_prior_norm = P_prior_norm[..., :eff_len]

        dist = torch.norm(P_u_norm - P_prior_norm, p=2, dim=-1, keepdim=True)
        adaptive_scale = torch.tanh(dist * 5.0)
        gamma = self.base_gamma + (self.max_gamma - self.base_gamma) * adaptive_scale

        P_pert_norm = (1 - gamma) * P_u_norm + gamma * P_prior_norm
        P_pert = P_pert_norm * E_u

        maps = self._get_coordinate_maps(h_freq.shape[2], h_freq.shape[3], h_freq.device)
        bin_map = maps["radius"] * self.num_sectors + maps["angle"]
        bin_map = torch.clamp(bin_map, max=eff_len - 1)

        flat_indices = bin_map.view(-1)
        Amp_pert_shifted = P_pert[:, :, flat_indices].view(h_freq.shape)
        Amp_pert = torch.fft.ifftshift(Amp_pert_shifted, dim=(-2, -1))

        h_freq_new = torch.fft.ifft2(Amp_pert * torch.exp(1j * fft_phase), norm="ortho").real
        pyramid[0] = h_freq_new
        img_harmonized = self._reconstruct_pyramid(pyramid)

        return self.normalize(img_harmonized.clamp(0, 1))
