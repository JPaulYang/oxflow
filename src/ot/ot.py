"""
Optimal Transport algorithms for point cloud matching.
"""

import torch
import numpy as np
from scipy.spatial.distance import cdist

try:
    import ot  # Python Optimal Transport library
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False


def optimal_transport_matching(pc1, pc2, cost_type='l2'):
    """
    Find optimal matching between two point clouds using optimal transport.

    Args:
        pc1: Tensor of shape (N1, 2) - first point cloud with N1 points
        pc2: Tensor of shape (N2, 2) - second point cloud with N2 points
             N1 and N2 can be different!
        cost_type: Type of cost function
                  - 'l2': L2 distance (default)
                  - 'l2_squared': squared L2 distance

    Returns:
        dict containing:
            - 'transport_plan': (N1, N2) numpy array - optimal transport plan matrix
                               transport_plan[i, j] = mass transported from point i to point j
            - 'cost_matrix': (N1, N2) numpy array - pairwise cost matrix
            - 'total_cost': float - total transport cost
            - 'matched_pairs': list of (i, j, mass) tuples for visualization

    Example:
        pc1 = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])  # 3 points
        pc2 = torch.tensor([[12.0, 12.0], [25.0, 25.0]])  # 2 points (different size!)
        result = optimal_transport_matching(pc1, pc2)

        # Access results
        plan = result['transport_plan']  # shape: (3, 2)
        cost = result['total_cost']       # total cost value
        pairs = result['matched_pairs']   # significant matchings

    Note:
        - Requires POT library: pip install POT
        - Uses uniform distribution (each point has equal mass)
        - Total mass is normalized to 1.0 for both point clouds
    """
    if not POT_AVAILABLE:
        raise ImportError("POT library required. Install with: pip install POT")

    if pc1.numel() == 0 or pc2.numel() == 0:
        raise ValueError("Point clouds cannot be empty")

    # Convert to numpy
    pc1_np = pc1.cpu().numpy().astype(np.float64)
    pc2_np = pc2.cpu().numpy().astype(np.float64)

    N1, N2 = pc1_np.shape[0], pc2_np.shape[0]

    # Compute cost matrix (pairwise distances) using vectorized cdist
    if cost_type == 'l2':
        C = cdist(pc1_np, pc2_np, metric='euclidean')
    elif cost_type == 'l2_squared':
        C = cdist(pc1_np, pc2_np, metric='sqeuclidean')
    else:
        raise ValueError(f"Unknown cost_type: {cost_type}")

    # Define uniform distributions (each point has equal mass)
    # Sum of masses = 1.0 for each distribution
    a = np.ones(N1) / N1  # source distribution (uniform)
    b = np.ones(N2) / N2  # target distribution (uniform)

    # Solve optimal transport problem
    # Returns transport plan T where T[i,j] = mass from i to j
    transport_plan = ot.emd(a, b, C)

    # Compute total cost
    total_cost = np.sum(transport_plan * C)

    # Extract significant matched pairs (mass > threshold) using vectorized operation
    threshold = 1e-6
    indices = np.argwhere(transport_plan > threshold)
    matched_pairs = [(i, j, transport_plan[i, j]) for i, j in indices]

    return {
        'transport_plan': transport_plan,
        'cost_matrix': C,
        'total_cost': total_cost,
        'matched_pairs': matched_pairs
    }

def match_pointcloud_counts_by_duplication(pc1, pc2):
    """
    Match point cloud counts by upsampling the smaller to match the larger.
    The smaller point cloud is duplicated cyclically to reach max(N1, N2).

    Args:
        pc1: Tensor of shape (N1, 2) - first point cloud
        pc2: Tensor of shape (N2, 2) - second point cloud

    Returns:
        Tuple of (pc1_matched, pc2_matched) where both have shape (max(N1, N2), 2)
    """
    N1, N2 = len(pc1), len(pc2)
    assert N1 > 0 and N2 > 0, "Point clouds must not be empty"

    if N1 == N2:
        return pc1, pc2

    N_max = max(N1, N2)

    if N1 < N_max:
        indices = [i % N1 for i in range(N_max)]
        pc1 = pc1[indices]

    if N2 < N_max:
        indices = [i % N2 for i in range(N_max)]
        pc2 = pc2[indices]

    return pc1, pc2

def reorder_by_optimal_transport(pc1, pc2, transport_plan):
    """
    Reorder pc2 according to optimal transport matching from pc1 to pc2.
    Since point counts are equal (N1 = N2 = N), we find the best matching.

    This extracts an approximate Monge map from the Kantorovich coupling
    using a greedy approach.

    Args:
        pc1: Tensor of shape (N, 2) - source point cloud
        pc2: Tensor of shape (N, 2) - target point cloud
        transport_plan: numpy array of shape (N, N) - OT transport plan

    Returns:
        Tensor of shape (N, 2) - pc2 reordered to match pc1
    """
    assert pc1.shape[0] == pc2.shape[0], "Point clouds must have the same number of points"
    N = len(pc1)

    # For each source point i, find the target point j with maximum transport mass
    matching = []
    used_targets = set()

    # Greedy matching: for each source, pick its best target
    for i in range(N):
        # Find target with maximum transport from source i
        available_mask = np.ones(N, dtype=bool)
        available_mask[list(used_targets)] = False

        # Among available targets, find the one with max transport from i
        masked_plan = transport_plan[i].copy()
        masked_plan[~available_mask] = -1

        j = np.argmax(masked_plan)
        matching.append(j)
        used_targets.add(j)

    # Reorder pc2 according to matching
    pc2_np = pc2.cpu().numpy()
    pc2_reordered = pc2_np[matching]

    return torch.from_numpy(pc2_reordered).to(pc2.device)
