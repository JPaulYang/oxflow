"""
Functions for converting between masks and point clouds.
"""

import torch
import numpy as np

try:
    from scipy.ndimage import binary_erosion, binary_fill_holes
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def mask_to_pointcloud(mask):
    """
    Convert binary mask to point cloud (all coordinates where mask == 1).

    Args:
        mask: Tensor of shape (T, 1, H, W) or (1, H, W) or (H, W)
              Binary mask with values 0 or 1

    Returns:
        - If input is (T, 1, H, W): List of T tensors, each (N_i, 2)
        - If input is (1, H, W) or (H, W): Single tensor (N, 2)
        Each point is [y, x] coordinates (row, column)
    """
    if mask.dim() == 4:  # (T, 1, H, W)
        T = mask.shape[0]
        pointclouds = []
        for t in range(T):
            mask_2d = mask[t, 0]  # (H, W)
            coords = torch.nonzero(mask_2d, as_tuple=False)  # (N, 2)
            pointclouds.append(coords)
        return pointclouds

    elif mask.dim() == 3:  # (1, H, W)
        mask_2d = mask[0]
        return torch.nonzero(mask_2d, as_tuple=False)

    elif mask.dim() == 2:  # (H, W)
        return torch.nonzero(mask, as_tuple=False)

    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")


def pointcloud_to_mask(pointcloud, H, W, closing_radius=3, smooth_sigma=0,
                       convex_hull=False):
    """
    Convert point cloud (with float coordinates) to binary mask.

    Args:
        pointcloud: Tensor of shape (N, 2) with [y, x] coordinates
        H, W: Height and width of output mask
        closing_radius: radius of disk structuring element for closing (0 to disable).
                        Ignored when convex_hull=True.
        smooth_sigma: Gaussian sigma for edge smoothing (0 to disable).
                      Ignored when convex_hull=True.
        convex_hull: If True, use convex hull of the point cloud instead of
                     per-pixel rounding + morphological closing. Produces
                     perfectly smooth boundaries.

    Returns:
        Binary mask tensor of shape (H, W)
    """
    mask = torch.zeros(H, W)

    if pointcloud.numel() == 0:
        return mask

    if convex_hull:
        from scipy.spatial import ConvexHull
        from skimage.draw import polygon

        pts = pointcloud.cpu().numpy() if hasattr(pointcloud, 'numpy') else np.array(pointcloud)
        pts = pts.astype(np.float64)

        if len(pts) < 3:
            return mask

        try:
            hull = ConvexHull(pts)
        except Exception:
            # Degenerate case (collinear points etc.)
            y = np.clip(np.round(pts[:, 0]).astype(int), 0, H - 1)
            x = np.clip(np.round(pts[:, 1]).astype(int), 0, W - 1)
            mask_np = np.zeros((H, W), dtype=np.float32)
            mask_np[y, x] = 1
            return torch.from_numpy(mask_np)

        hull_pts = pts[hull.vertices]  # ordered hull vertices
        rr, cc = polygon(hull_pts[:, 0], hull_pts[:, 1], shape=(H, W))
        mask_np = np.zeros((H, W), dtype=np.float32)
        mask_np[rr, cc] = 1
        return torch.from_numpy(mask_np)

    # ── Default path: per-pixel rounding + morphological ops ──────────
    # Round to nearest pixel
    y = pointcloud[:, 0].round().long()
    x = pointcloud[:, 1].round().long()

    # Clamp to valid range
    y = y.clamp(0, H - 1)
    x = x.clamp(0, W - 1)

    mask[y, x] = 1

    # Morphological closing: fill gaps while preserving shape (including concavities)
    if closing_radius > 0 and SCIPY_AVAILABLE:
        from skimage.morphology import disk, closing
        mask_np = mask.numpy().astype(bool)
        mask_np = closing(mask_np, footprint=disk(closing_radius))
        mask_np = binary_fill_holes(mask_np)
        mask = torch.from_numpy(mask_np.astype(np.float32))

    # Edge smoothing: Gaussian blur + re-threshold removes small protrusions
    if smooth_sigma > 0 and SCIPY_AVAILABLE:
        from scipy.ndimage import gaussian_filter
        mask_np = mask.numpy().astype(np.float64)
        mask_np = gaussian_filter(mask_np, sigma=smooth_sigma)
        mask_np = (mask_np > 0.5).astype(np.float32)
        mask_np = binary_fill_holes(mask_np).astype(np.float32)
        mask = torch.from_numpy(mask_np)

    return mask


def extract_boundary_points(mask_2d):
    """
    Extract boundary pixels from a 2D binary mask.

    Args:
        mask_2d: Tensor of shape (H, W) - binary mask

    Returns:
        Tensor of shape (N, 2) - boundary pixel coordinates [y, x]
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for boundary extraction. Install with: pip install scipy")

    # Convert to numpy for scipy
    mask_np = mask_2d.cpu().numpy().astype(bool)

    # Erode the mask by 1 pixel
    eroded = binary_erosion(mask_np, structure=np.ones((3, 3)))

    # Boundary = original mask - eroded mask
    boundary_np = mask_np & (~eroded)

    # Convert back to torch and get coordinates
    boundary_coords = torch.from_numpy(np.argwhere(boundary_np))  # Returns [y, x] format

    return boundary_coords


def mask_to_pointcloud_boundary(mask):
    """
    Convert binary masks to boundary point clouds only (no interior points).
    Point counts will naturally differ based on mask perimeter.

    Args:
        mask: Tensor of shape (T, 1, H, W) - T timepoints
              Binary mask with values 0 or 1

    Returns:
        List of T tensors, each of shape (N_i, 2) where N_i can be DIFFERENT
        Each point is [y, x] coordinates (row, column) of boundary pixels

    Example:
        mask = torch.zeros(3, 1, 256, 256)  # 3 timepoints
        mask[0, 0, 100:120, 100:120] = 1    # Small mask
        mask[1, 0, 100:150, 100:150] = 1    # Large mask
        pointclouds = mask_to_pointcloud_boundary(mask)
        # pointclouds[0] might have 80 points (small perimeter)
        # pointclouds[1] might have 200 points (large perimeter)
    """

    if mask.dim() != 4:
        raise ValueError(f"Expected 4D input (T, 1, H, W), got shape {mask.shape}")

    T = mask.shape[0]

    # Extract boundaries for all timepoints
    boundary_pointclouds = []

    for t in range(T):
        mask_2d = mask[t, 0]  # (H, W)
        boundary = extract_boundary_points(mask_2d)
        boundary_pointclouds.append(boundary)

    return boundary_pointclouds

def pointcloud_to_mask_boundary_fill(pointcloud, shape, k_neighbors=2):
    """
    Convert boundary point cloud to mask by connecting each point to its k nearest neighbors.

    1. Round points to integer pixel coordinates
    2. Each point connects to its k nearest neighbors with lines
    3. Fill the interior using binary_fill_holes

    Args:
        pointcloud: (N, 2) tensor or array, [y, x] coordinates (can be float)
        shape: (H, W) output mask shape
        k_neighbors: Number of nearest neighbors to connect to (default 2)

    Returns:
        (H, W) binary mask tensor
    """
    from skimage.draw import line
    from scipy.spatial.distance import cdist

    H, W = shape
    points = pointcloud.numpy() if hasattr(pointcloud, 'numpy') else pointcloud

    if len(points) < 3:
        return torch.zeros(shape)

    # Round to integer coordinates
    coords = np.column_stack([
        np.clip(np.round(points[:, 0]).astype(int), 0, H-1),
        np.clip(np.round(points[:, 1]).astype(int), 0, W-1)
    ])

    # Compute distance matrix and find k nearest neighbors for each point
    dist_matrix = cdist(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)

    # Get indices of k nearest neighbors for each point
    k = min(k_neighbors, len(coords) - 1)
    nearest_k_indices = np.argsort(dist_matrix, axis=1)[:, :k]

    # Create boundary by connecting each point to its k nearest neighbors
    boundary = np.zeros(shape, dtype=bool)

    for i in range(len(coords)):
        y0, x0 = coords[i]
        for j in nearest_k_indices[i]:
            y1, x1 = coords[j]
            rr, cc = line(y0, x0, y1, x1)
            boundary[rr, cc] = True

    # Fill the interior
    filled = binary_fill_holes(boundary)

    return torch.from_numpy(filled.astype(np.float32))