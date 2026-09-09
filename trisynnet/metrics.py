"""Evaluation metrics: region overlap (Dice/IoU/Precision/Recall) and
boundary-specific metrics (HD95, ASSD, Boundary IoU, Boundary F1)."""

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def compute_per_sample_metrics(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    target_flat = target_binary.view(target_binary.size(0), -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)

    dice = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
    iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
    precision = (tp + 1e-8) / (tp + fp + 1e-8)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)

    return {"Dice": dice, "IoU": iou, "Precision": precision, "Recall": recall}


def _get_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return mask & (~eroded)


def hd95_assd(pred: np.ndarray, gt: np.ndarray):
    """Returns (HD95, ASSD) in pixels.

    Both empty -> (0, 0). Exactly one empty -> (nan, nan); callers should
    exclude these from the mean and report how many images were excluded.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan"), float("nan")

    pred_b = _get_boundary(pred)
    gt_b = _get_boundary(gt)

    dt_gt = distance_transform_edt(~gt_b)
    dt_pred = distance_transform_edt(~pred_b)

    d_pred_to_gt = dt_gt[pred_b]
    d_gt_to_pred = dt_pred[gt_b]

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95)), float(all_d.mean())


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02):
    """Boundary IoU (Cheng et al., CVPR 2021), with a dilation band scaled
    to image diagonal so it is independent of polyp size."""
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    H, W = gt.shape
    img_diag = np.sqrt(H ** 2 + W ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))

    def _band(mask):
        eroded = binary_erosion(mask.astype(bool), structure=np.ones((3, 3)),
                                 iterations=dilation, border_value=1)
        return mask.astype(bool) & (~eroded)

    pred_band = _band(pred)
    gt_band = _band(gt)

    if pred_band.sum() == 0 and gt_band.sum() == 0:
        return 1.0
    inter = np.logical_and(pred_band, gt_band).sum()
    union = np.logical_or(pred_band, gt_band).sum()
    return float(inter) / float(union) if union else 0.0


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance_px: int = 2):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    pred_b = _get_boundary(pred)
    gt_b = _get_boundary(gt)
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0

    dt_gt = distance_transform_edt(~gt_b)
    dt_pred = distance_transform_edt(~pred_b)

    precision = (dt_gt[pred_b] <= tolerance_px).sum() / pred_b.sum()
    recall = (dt_pred[gt_b] <= tolerance_px).sum() / gt_b.sum()

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_boundary_metrics_batch(pred_prob, target, threshold=0.5,
                                    dilation_ratio=0.02, boundary_tolerance_px=2):
    pred_bin = (pred_prob > threshold).float()
    B = pred_bin.shape[0]

    pred_np = pred_bin.squeeze(1).detach().cpu().numpy().astype(bool)
    gt_np = (target.squeeze(1).detach().cpu().numpy() > 0.5)

    hd95_list, assd_list, biou_list, bf1_list = [], [], [], []
    for i in range(B):
        hd95, assd = hd95_assd(pred_np[i], gt_np[i])
        biou_list.append(boundary_iou(pred_np[i], gt_np[i], dilation_ratio=dilation_ratio))
        bf1_list.append(boundary_f1(pred_np[i], gt_np[i], tolerance_px=boundary_tolerance_px))
        hd95_list.append(hd95)
        assd_list.append(assd)

    return {
        "HD95": torch.tensor(hd95_list, dtype=torch.float32),
        "ASSD": torch.tensor(assd_list, dtype=torch.float32),
        "BoundaryIoU": torch.tensor(biou_list, dtype=torch.float32),
        "BoundaryF1": torch.tensor(bf1_list, dtype=torch.float32),
    }


def aggregate_boundary_metrics(list_of_batches):
    out = {}
    for name in ["HD95", "ASSD", "BoundaryIoU", "BoundaryF1"]:
        vals = torch.cat([b[name] for b in list_of_batches])
        n_total = vals.numel()
        valid = vals[~torch.isnan(vals)]
        n_excluded = n_total - valid.numel()
        out[name] = {
            "mean": float(valid.mean()) if valid.numel() > 0 else float("nan"),
            "std": float(valid.std()) if valid.numel() > 1 else 0.0,
            "n_total": n_total,
            "n_excluded_nan": n_excluded,
        }
    return out
