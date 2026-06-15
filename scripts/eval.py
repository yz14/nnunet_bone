""" 1. Surface Dice>=95% (误差三个像素以内) 2. Dice >=95%"""

"""
骨头分割评价指标 (BBox加速版)
Metrics: 1. Dice  2. Surface Dice (NSD, tolerance=3 pixels)
优化: 先用 GT+Pred 的联合 bbox 截取 ROI，再计算指标，大幅减少 EDT 计算量
"""

import os
import glob
import time
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt


# ─────────────────────────── bbox crop ───────────────────────────────────────

def get_bbox(mask: np.ndarray, pad: int = 3) -> tuple:
    """
    计算非零区域的 bounding box，并向外 pad 若干像素（至少容纳 tolerance）。
    返回 (z0,z1, y0,y1, x0,x1) 切片索引。
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        # 全空则返回整个体积
        return (0, mask.shape[0], 0, mask.shape[1], 0, mask.shape[2])

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1   # +1 → exclusive

    # 加 padding，防止表面体素距离计算被边界截断
    z0 = max(z0 - pad, 0);  z1 = min(z1 + pad, mask.shape[0])
    y0 = max(y0 - pad, 0);  y1 = min(y1 + pad, mask.shape[1])
    x0 = max(x0 - pad, 0);  x1 = min(x1 + pad, mask.shape[2])

    return (z0, z1, y0, y1, x0, x1)


def crop_to_bbox(pred: np.ndarray, gt: np.ndarray,
                 pad: int = 3) -> tuple:
    """
    用 pred | gt 的联合 bbox 同时裁剪两个 mask，返回裁剪后的数组和 bbox 信息。
    pad 应 >= tolerance，保证表面 EDT 不被边界误截。
    """
    union = (pred > 0) | (gt > 0)
    z0, z1, y0, y1, x0, x1 = get_bbox(union, pad=pad)

    pred_crop = pred[z0:z1, y0:y1, x0:x1]
    gt_crop   = gt  [z0:z1, y0:y1, x0:x1]

    info = {
        "full_shape" : pred.shape,
        "crop_shape" : pred_crop.shape,
        "bbox"       : (z0, z1, y0, y1, x0, x1),
        "ratio"      : pred_crop.size / pred.size,
    }
    return pred_crop, gt_crop, info


# ─────────────────────────── core metrics ────────────────────────────────────

def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def get_surface_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    eroded = binary_erosion(mask, border_value=0)
    return mask & ~eroded


def compute_surface_dice(pred: np.ndarray, gt: np.ndarray,
                         tolerance: float = 3.0) -> float:
    """
    NSD = (|S_pred ∩ B_gt^τ| + |S_gt ∩ B_pred^τ|) / (|S_pred| + |S_gt|)
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    surf_pred = get_surface_mask(pred)
    surf_gt   = get_surface_mask(gt)

    n_surf_pred = surf_pred.sum()
    n_surf_gt   = surf_gt.sum()

    if n_surf_pred == 0 and n_surf_gt == 0:
        return 1.0
    if n_surf_pred == 0 or n_surf_gt == 0:
        return 0.0

    dist_pred = distance_transform_edt(~surf_pred)
    dist_gt   = distance_transform_edt(~surf_gt)

    pred_in_band = surf_pred & (dist_gt   <= tolerance)
    gt_in_band   = surf_gt   & (dist_pred <= tolerance)

    return float((pred_in_band.sum() + gt_in_band.sum()) / (n_surf_pred + n_surf_gt))


# ─────────────────────────── I/O helpers ─────────────────────────────────────

def load_nii_binary(path: str) -> np.ndarray:
    img = nib.load(path)
    return (np.asarray(img.dataobj) > 0).astype(np.uint8)


def find_case_ids(pred_dir: str) -> list:
    files = glob.glob(os.path.join(pred_dir, "*.nii.gz"))
    ids = [os.path.basename(f).replace(".nii.gz", "") for f in files]
    return sorted(ids)


# ─────────────────────────── main evaluation ─────────────────────────────────

def evaluate(pred_dir: str, gt_dir: str,
             tolerance: float = 3.0,
             use_bbox: bool = True) -> pd.DataFrame:
    """
    Parameters
    ----------
    use_bbox  : 是否用 bbox 裁剪加速（推荐 True）
    tolerance : NSD 容忍距离，单位像素（默认 3）
                pad 会自动设为 tolerance+1，保证边界安全
    """
    case_ids = find_case_ids(pred_dir)
    if not case_ids:
        raise FileNotFoundError(f"No prediction .nii.gz found in: {pred_dir}")

    pad = int(tolerance) + 1      # bbox padding 略大于 tolerance
    records = []
    total_t = 0.0

    for cid in case_ids:
        pred_path = os.path.join(pred_dir, f"{cid}.nii.gz")
        gt_path   = os.path.join(gt_dir,   f"{cid}-seg.nii.gz")

        if not os.path.exists(gt_path):
            print(f"[WARN] GT not found for case {cid}")
            continue

        t0   = time.perf_counter()
        pred = load_nii_binary(pred_path)
        gt   = load_nii_binary(gt_path)

        if pred.shape != gt.shape:
            print(f"[WARN] Shape mismatch case {cid}: pred {pred.shape} vs gt {gt.shape}")
            continue

        crop_info = None
        if use_bbox:
            pred, gt, crop_info = crop_to_bbox(pred, gt, pad=pad)

        dice = compute_dice(pred, gt)
        nsd  = compute_surface_dice(pred, gt, tolerance=tolerance)
        elapsed = time.perf_counter() - t0
        total_t += elapsed

        pass_dice = dice >= 0.95
        pass_nsd  = nsd  >= 0.95
        status    = "✓" if (pass_dice and pass_nsd) else "✗"

        crop_str = ""
        if crop_info:
            r = crop_info["ratio"]
            crop_str = f"  crop={crop_info['crop_shape']}  ({r*100:.1f}% of full)"

        print(f"[{status}] Case {cid:>3s} | Dice={dice:.4f} | NSD={nsd:.4f} "
              f"| {elapsed:.2f}s{crop_str}")

        records.append({
            "case_id"     : cid,
            "Dice"        : round(dice, 4),
            "SurfaceDice" : round(nsd,  4),
            "Dice≥95%"    : pass_dice,
            "NSD≥95%"     : pass_nsd,
            "Both≥95%"    : pass_dice and pass_nsd,
            "time_s"      : round(elapsed, 3),
            "crop_ratio"  : round(crop_info["ratio"], 4) if crop_info else 1.0,
        })

    df = pd.DataFrame(records)
    print(f"\n总耗时: {total_t:.1f}s  (平均 {total_t/len(df):.2f}s/case)")
    return df


def print_summary(df: pd.DataFrame, tolerance: float) -> None:
    n = len(df)
    print("\n" + "=" * 60)
    print(f"  评价汇总  (tolerance={tolerance:.0f}px | bbox加速)")
    print("=" * 60)
    print(f"  总病例数          : {n}")
    print(f"  Mean Dice         : {df['Dice'].mean():.4f} ± {df['Dice'].std():.4f}")
    print(f"  Mean Surface Dice : {df['SurfaceDice'].mean():.4f} ± {df['SurfaceDice'].std():.4f}")
    print(f"  Dice   ≥ 95%      : {df['Dice≥95%'].sum()}/{n}  ({100*df['Dice≥95%'].mean():.1f}%)")
    print(f"  NSD    ≥ 95%      : {df['NSD≥95%'].sum()}/{n}  ({100*df['NSD≥95%'].mean():.1f}%)")
    print(f"  Both   ≥ 95%      : {df['Both≥95%'].sum()}/{n}  ({100*df['Both≥95%'].mean():.1f}%)")
    if "crop_ratio" in df.columns:
        print(f"  平均 crop 体积比   : {df['crop_ratio'].mean()*100:.1f}%  "
              f"(EDT 计算量减少 {(1-df['crop_ratio'].mean())*100:.1f}%)")
    print("=" * 60)

    failed = df[~df["Both≥95%"]]
    if not failed.empty:
        print("\n  未达标病例:")
        for _, row in failed.iterrows():
            print(f"    case {row['case_id']:>3s} | Dice={row['Dice']:.4f} | NSD={row['SurfaceDice']:.4f}")
    else:
        print("\n  所有病例均达标 ✓")
    print()


# ─────────────────────────── entry point ─────────────────────────────────────
if __name__ == "__main__":
    pred_path = "/data0/yzhen/data/bone_data_test_p4"
    gt_path   = "/data0/yzhen/data/bone_data_test"
    TOLERANCE = 3.0

    df = evaluate(pred_path, gt_path, tolerance=TOLERANCE, use_bbox=True)
    print_summary(df, TOLERANCE)

    out_csv = os.path.join(pred_path, "evaluation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"结果已保存: {out_csv}")