import torch
import numpy as np
from src.ot.mask_to_pointcloud import (
    mask_to_pointcloud,
    pointcloud_to_mask
)
from src.ot.ot import (
    optimal_transport_matching,
    match_pointcloud_counts_by_duplication,
    reorder_by_optimal_transport
)

def interpolate_pointclouds(pc1, pc2, t):
    """Linear interpolation between two aligned point clouds."""
    return (1 - t) * pc1.float() + t * pc2.float()

def compute_ot_interpolated_mask(mask_t0, mask_t1, t=0.5, smooth_sigma=0,
                                  convex_hull=False):
    """
    Compute OT-interpolated mask between two timepoints using full mask point clouds.

    Args:
        mask_t0: (1, H, W) tensor - mask at first timepoint
        mask_t1: (1, H, W) tensor - mask at last timepoint
        t: interpolation parameter (0=t0, 1=t1)
        smooth_sigma: Gaussian sigma for edge smoothing (0=off)
        convex_hull: Use convex hull for perfectly smooth boundaries (for viz)

    Returns:
        (1, H, W) tensor - interpolated mask
    """
    H, W = mask_t0.shape[1], mask_t0.shape[2]

    # Stack masks for point cloud extraction: (2, 1, H, W)
    masks_stacked = torch.stack([mask_t0, mask_t1], dim=0)

    # Extract full mask point clouds
    full_pcs = mask_to_pointcloud(masks_stacked)
    pc0, pc1 = full_pcs[0], full_pcs[1]

    if len(pc0) < 3 or len(pc1) < 3:
        print("  Warning: Not enough points, returning t0 mask")
        return mask_t0

    # Match point cloud counts (adjust pc1 to match pc0)
    pc0_matched, pc1_matched = match_pointcloud_counts_by_duplication(pc0, pc1)

    # Compute OT matching
    ot_result = optimal_transport_matching(pc0_matched, pc1_matched)

    # Reorder pc1 according to OT plan
    pc1_reordered = reorder_by_optimal_transport(
        pc0_matched, pc1_matched, ot_result['transport_plan']
    )

    # Interpolate point clouds
    pc_interp = interpolate_pointclouds(pc0_matched, pc1_reordered, t)

    # Convert back to mask
    mask_interp = pointcloud_to_mask(pc_interp, H, W, smooth_sigma=smooth_sigma,
                                     convex_hull=convex_hull)

    return mask_interp.unsqueeze(0)  # (1, H, W)


def compute_random_interpolated_mask(mask_t0, mask_t1, t=0.5, smooth_sigma=0,
                                      convex_hull=False):
    """
    Compute interpolated mask using random point cloud matching (ablation baseline).

    Same pipeline as OT interpolation but with random permutation instead of
    optimal transport matching. This isolates the contribution of OT's optimal
    matching strategy.

    Args:
        mask_t0: (1, H, W) tensor - mask at first timepoint
        mask_t1: (1, H, W) tensor - mask at last timepoint
        t: interpolation parameter (0=t0, 1=t1)
        smooth_sigma: Gaussian sigma for edge smoothing (0=off)
        convex_hull: Use convex hull for perfectly smooth boundaries (for viz)

    Returns:
        (1, H, W) tensor - interpolated mask
    """
    H, W = mask_t0.shape[1], mask_t0.shape[2]

    masks_stacked = torch.stack([mask_t0, mask_t1], dim=0)
    full_pcs = mask_to_pointcloud(masks_stacked)
    pc0, pc1 = full_pcs[0], full_pcs[1]

    if len(pc0) < 3 or len(pc1) < 3:
        print("  Warning: Not enough points, returning t0 mask")
        return mask_t0

    pc0_matched, pc1_matched = match_pointcloud_counts_by_duplication(pc0, pc1)

    # Random permutation instead of OT matching
    perm = np.random.permutation(len(pc1_matched))
    pc1_reordered = pc1_matched[perm]

    pc_interp = interpolate_pointclouds(pc0_matched, pc1_reordered, t)
    mask_interp = pointcloud_to_mask(pc_interp, H, W, smooth_sigma=smooth_sigma,
                                     convex_hull=convex_hull)

    return mask_interp.unsqueeze(0)  # (1, H, W)
