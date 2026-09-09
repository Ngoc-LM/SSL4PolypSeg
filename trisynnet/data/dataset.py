"""Dataset and dataloader construction for semi-supervised polyp
segmentation, expecting a single .npz archive with train/val/test image and
mask arrays (see README for the expected keys and layout)."""

import random

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


def worker_init_fn(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Weak augmentation: geometric only, applied to both the "weak" (teacher)
# and as the base for the "strong" (student) view.
weak_geometric = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
])

# Strong augmentation: color/noise only, applied on top of the weak view.
strong_color = A.Compose([
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
])

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

normalize_transform = A.Compose([
    A.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ToTensorV2(),
])


class SemiSupervisedPolypDS(Dataset):
    """Reads `{type}_img`/`{type}_msk` arrays from an .npz archive.

    Train mode returns (weak_view, strong_view, mask, dummy_label).
    Val/test mode returns (normalized_image, mask), resized to `val_size`.
    """

    def __init__(self, data_path, type="train", labeled_indices=None,
                 img_size=256, val_size=352):
        self.data_path = data_path
        self.type = type
        self.is_train = type == "train"
        self.img_size = img_size
        self.val_size = val_size

        data = np.load(data_path, allow_pickle=False)
        self.images = data[f"{type}_img"]  # BGR
        self.masks = data[f"{type}_msk"]

        if self.masks.ndim == 4:
            self.masks = self.masks.squeeze(-1)
        if self.masks.max() > 1:
            self.masks = self.masks / 255.0
        self.masks = np.clip(self.masks, 0, 1).astype(np.float32)

        self.total_samples = len(self.images)
        self.indices = labeled_indices if labeled_indices is not None else list(range(self.total_samples))

        self.resize_train = A.Resize(img_size, img_size)
        self.resize_val = A.Resize(val_size, val_size)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]

        img_bgr = self.images[global_idx].copy()
        msk = self.masks[global_idx].copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if not self.is_train:
            aug = self.resize_val(image=img_rgb, mask=msk)
            normalized = normalize_transform(image=aug["image"])["image"]
            return normalized, torch.from_numpy(aug["mask"]).float().unsqueeze(0)

        resized = self.resize_train(image=img_rgb, mask=msk)
        img_base, msk_base = resized["image"], resized["mask"]

        weak_aug = weak_geometric(image=img_base, mask=msk_base)
        img_weak, mask_final = weak_aug["image"], weak_aug["mask"]

        strong_aug = strong_color(image=img_weak)
        img_strong = strong_aug["image"]

        weak_view = normalize_transform(image=img_weak)["image"]
        strong_view = normalize_transform(image=img_strong)["image"]
        mask_t = torch.from_numpy(mask_final).float().unsqueeze(0)

        return weak_view, strong_view, mask_t, torch.tensor(1.0)


def build_semisup_loaders(data_path, batch_train=16, batch_val=8, labeled_ratio=0.1,
                           img_size=256, val_size=352, num_workers=4, seed=42):
    """Splits the `train` split into a labeled/unlabeled subset (by
    `labeled_ratio`, seeded), and builds a held-in `val` loader plus any
    `test_*` loaders present in the archive.

    Returns: ({"labeled": DataLoader, "unlabeled": DataLoader}, val_loader,
    {test_set_name: DataLoader}).
    """
    data = np.load(data_path, allow_pickle=False)
    total_samples = len(data["train_img"])

    indices = np.arange(total_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_labeled = max(1, int(total_samples * labeled_ratio))
    labeled_idxs = indices[:n_labeled].tolist()
    unlabeled_idxs = indices[n_labeled:].tolist()
    if len(unlabeled_idxs) == 0:
        unlabeled_idxs = labeled_idxs

    train_labeled_ds = SemiSupervisedPolypDS(data_path, "train", labeled_indices=labeled_idxs, img_size=img_size)
    train_unlabeled_ds = SemiSupervisedPolypDS(data_path, "train", labeled_indices=unlabeled_idxs, img_size=img_size)
    val_ds = SemiSupervisedPolypDS(data_path, "val", img_size=img_size, val_size=val_size)

    n_labeled_batch = batch_train // 2
    n_unlabeled_batch = batch_train - n_labeled_batch

    loader_args = dict(num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    label_loader = DataLoader(train_labeled_ds, batch_size=n_labeled_batch, shuffle=True, drop_last=True, **loader_args)
    unlabel_loader = DataLoader(train_unlabeled_ds, batch_size=n_unlabeled_batch, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=batch_val, shuffle=False, **loader_args)

    test_loaders = {}
    test_sets = ["test_kvasir", "test_etis", "test_cvc300", "test_clinic", "test_colon"]
    try:
        keys = data.files
        for t in test_sets:
            if f"{t}_img" in keys:
                test_ds = SemiSupervisedPolypDS(data_path, type=t, img_size=img_size, val_size=val_size)
                test_loaders[t] = DataLoader(test_ds, batch_size=4, shuffle=False, **loader_args)
    except Exception as e:
        print(f"Warning checking test keys: {e}")

    return {"labeled": label_loader, "unlabeled": unlabel_loader}, val_loader, test_loaders
