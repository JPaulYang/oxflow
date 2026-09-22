"""
Compute trajectory smoothness metrics for OT vs Linear vs Random mask interpolation.

For each (patient, slice) with >=2 tumor timepoints, interpolates masks at
alpha in {0.25, 0.5, 0.75} using three methods (OT, Linear, Random) and
computes:
  1. Volume monotonicity violation rate (%)
  2. Boundary perimeter total variation (normalized)
  3. Center-of-mass path curvature (radians)

Output: CSV + LaTeX table fragment + optional volume-vs-alpha plot
(mean ± std shaded band across all valid trajectories).
When --plot is used, also saves plot_data.json so the figure can be redrawn
without recomputing.

Usage:
    python script/eval/compute_trajectory_metrics.py \
        --data_path data_processed/longitudinal_data.pt \
        --output_dir outputs/trajectory_metrics

    # With visualization (mean ± std plot) + save plot data for replotting
    python script/eval/compute_trajectory_metrics.py \
        --data_path data_processed/longitudinal_data.pt \
        --output_dir outputs/trajectory_metrics \
        --plot
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ot.mask_to_pointcloud import mask_to_pointcloud, pointcloud_to_mask
from src.ot.ot import (
    optimal_transport_matching,
    match_pointcloud_counts_by_duplication,
    reorder_by_optimal_transport,
)
from src.ot.interpolation import interpolate_pointclouds


# ─── Point cloud subsampling ───────────────────────────────────────────────

MAX_POINTS = 500  # Cap point clouds at this size for tractable OT


def subsample_pointcloud(pc, max_points):
    """Randomly subsample a point cloud if it exceeds max_points."""
    if len(pc) <= max_points:
        return pc
    idx = np.random.choice(len(pc), max_points, replace=False)
    return pc[idx]


# ─── Mask interpolation: linear (pixel-space) ──────────────────────────────

def compute_linear_interpolated_mask(mask_t0, mask_t1, t=0.5):
    """Pixel-level linear interpolation with threshold at 0.5."""
    interp = (1 - t) * mask_t0.float() + t * mask_t1.float()
    return (interp > 0.5).float()


def compute_ot_interpolated_masks_batch(mask_t0, mask_t1, alphas, max_points=MAX_POINTS):
    """
    Compute OT-interpolated masks for multiple alphas at once.
    Performs OT matching only once, then interpolates for each alpha.
    Subsamples point clouds to max_points for speed.

    Returns list of (1, H, W) tensors, one per alpha.
    """
    H, W = mask_t0.shape[1], mask_t0.shape[2]
    masks_stacked = torch.stack([mask_t0, mask_t1], dim=0)
    full_pcs = mask_to_pointcloud(masks_stacked)
    pc0, pc1 = full_pcs[0], full_pcs[1]

    if len(pc0) < 3 or len(pc1) < 3:
        return [mask_t0 for _ in alphas]

    # Subsample before OT
    pc0 = subsample_pointcloud(pc0, max_points)
    pc1 = subsample_pointcloud(pc1, max_points)

    pc0_m, pc1_m = match_pointcloud_counts_by_duplication(pc0, pc1)
    ot_result = optimal_transport_matching(pc0_m, pc1_m)
    pc1_reordered = reorder_by_optimal_transport(pc0_m, pc1_m, ot_result['transport_plan'])

    results = []
    for alpha in alphas:
        pc_interp = interpolate_pointclouds(pc0_m, pc1_reordered, alpha)
        mask_interp = pointcloud_to_mask(pc_interp, H, W)
        results.append(mask_interp.unsqueeze(0))
    return results


def compute_random_interpolated_masks_batch(mask_t0, mask_t1, alphas, max_points=MAX_POINTS):
    """
    Compute random-matched interpolated masks for multiple alphas at once.
    Subsamples point clouds to max_points for speed.
    """
    H, W = mask_t0.shape[1], mask_t0.shape[2]
    masks_stacked = torch.stack([mask_t0, mask_t1], dim=0)
    full_pcs = mask_to_pointcloud(masks_stacked)
    pc0, pc1 = full_pcs[0], full_pcs[1]

    if len(pc0) < 3 or len(pc1) < 3:
        return [mask_t0 for _ in alphas]

    pc0 = subsample_pointcloud(pc0, max_points)
    pc1 = subsample_pointcloud(pc1, max_points)

    pc0_m, pc1_m = match_pointcloud_counts_by_duplication(pc0, pc1)
    perm = np.random.permutation(len(pc1_m))
    pc1_reordered = pc1_m[perm]

    results = []
    for alpha in alphas:
        pc_interp = interpolate_pointclouds(pc0_m, pc1_reordered, alpha)
        mask_interp = pointcloud_to_mask(pc_interp, H, W)
        results.append(mask_interp.unsqueeze(0))
    return results


# ─── Metric helpers ─────────────────────────────────────────────────────────

def mask_area(mask):
    """Foreground pixel count."""
    return mask.sum().item()


def mask_perimeter(mask):
    """Boundary pixel count via erosion."""
    m = mask.squeeze().numpy().astype(bool)
    if m.sum() == 0:
        return 0.0
    eroded = binary_erosion(m, structure=np.ones((3, 3)))
    return float((m & ~eroded).sum())


def mask_centroid(mask):
    """Center of mass as (y, x) array."""
    m = mask.squeeze()
    coords = torch.nonzero(m, as_tuple=False).float()
    if len(coords) == 0:
        return np.array([0.0, 0.0])
    return coords.mean(dim=0).numpy()


# ─── Trajectory-level metrics ───────────────────────────────────────────────

def volume_monotonicity_violation(volumes):
    """
    Check whether the volume sequence is monotone.

    Returns True if the trajectory violates monotonicity (i.e., the
    intermediate volumes oscillate rather than interpolating smoothly
    between the endpoints).
    """
    if len(volumes) < 3:
        return False

    # Expected direction: sign of (last - first)
    direction = volumes[-1] - volumes[0]

    if abs(direction) < 1:  # essentially same volume
        # Check for any large deviation from endpoints
        baseline = volumes[0]
        for v in volumes[1:-1]:
            if abs(v - baseline) > max(10, 0.05 * baseline):
                return True
        return False

    sign = np.sign(direction)
    # Check that each intermediate is between endpoints
    lo, hi = min(volumes[0], volumes[-1]), max(volumes[0], volumes[-1])
    for v in volumes[1:-1]:
        if v < lo - 1 or v > hi + 1:  # 1-pixel tolerance
            return True
    # Also check monotonicity of consecutive steps
    diffs = np.diff(volumes)
    signs = np.sign(diffs)
    nonzero_signs = signs[signs != 0]
    if len(nonzero_signs) > 0:
        # If there are sign changes in the differences, it's not monotone
        if not (np.all(nonzero_signs >= 0) or np.all(nonzero_signs <= 0)):
            return True
    return False


def perimeter_total_variation(perimeters):
    """Sum of |consecutive perimeter differences|, normalized by mean perimeter."""
    tv = float(np.sum(np.abs(np.diff(perimeters))))
    mean_perim = np.mean(perimeters)
    return tv / (mean_perim + 1e-6)


def centroid_curvature(centroids):
    """
    Sum of angular deflections at interior points of the centroid path.
    Lower = smoother trajectory.
    """
    if len(centroids) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(centroids) - 1):
        v1 = centroids[i] - centroids[i - 1]
        v2 = centroids[i + 1] - centroids[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        total += np.arccos(cos_angle)
    return float(total)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute trajectory smoothness metrics for mask interpolation methods')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to longitudinal_data.pt')
    parser.add_argument('--output_dir', type=str, default='outputs/trajectory_metrics',
                        help='Output directory')
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.1,0.15,0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,0.75,0.8,0.85,0.9,0.95],
                        help='Interpolation alphas (default: 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (for random matching method)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate volume-vs-alpha visualization')
    parser.add_argument('--n_plot', type=int, default=5,
                        help='(Deprecated, ignored) Previously limited trajectories to plot')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_path}")
    all_data = torch.load(args.data_path, weights_only=False)
    print(f"  Total entries: {len(all_data)}")

    # Collect valid trajectory pairs: (patient, slice) with >=2 tumor timepoints
    method_names = ['OT Mask', 'Linear Mask', 'Random Matching']

    # Per-method accumulators
    results = {m: {
        'mono_violations': [],  # bool per trajectory
        'perim_tv': [],         # float per trajectory
        'centroid_curv': [],    # float per trajectory
    } for m in method_names}

    # For optional plotting: store volume trajectories
    plot_data = {m: [] for m in method_names}

    n_valid_pairs = 0
    n_skipped = 0

    for item in tqdm(all_data, desc='Computing trajectory metrics'):
        has_tumor_per_tp = item.get('has_tumor_per_timepoint',
                                     [True] * item['ct_slices'].shape[0])
        masks = item['masks']  # (T, 1, H, W)

        # Find timepoints with tumor
        tumor_indices = [t for t, has in enumerate(has_tumor_per_tp) if has]
        if len(tumor_indices) < 2:
            continue

        # Iterate over consecutive pairs of tumor timepoints
        for pair_idx in range(len(tumor_indices) - 1):
            ti = tumor_indices[pair_idx]
            tj = tumor_indices[pair_idx + 1]

            mask_t0 = (masks[ti] > 0).float()  # (1, H, W)
            mask_t1 = (masks[tj] > 0).float()

            # Skip if either mask is too small
            if mask_area(mask_t0) < 10 or mask_area(mask_t1) < 10:
                n_skipped += 1
                continue

            n_valid_pairs += 1

            # Compute all alphas for each method at once (OT matching done once)
            try:
                ot_masks = compute_ot_interpolated_masks_batch(
                    mask_t0, mask_t1, args.alphas)
            except Exception:
                ot_masks = [mask_t0 for _ in args.alphas]

            linear_masks = [compute_linear_interpolated_mask(
                mask_t0, mask_t1, t=a) for a in args.alphas]

            try:
                random_masks = compute_random_interpolated_masks_batch(
                    mask_t0, mask_t1, args.alphas)
            except Exception:
                random_masks = [mask_t0 for _ in args.alphas]

            method_interps = {
                'OT Mask': ot_masks,
                'Linear Mask': linear_masks,
                'Random Matching': random_masks,
            }

            for method_name in method_names:
                # Full trajectory with endpoints (for volume monotonicity)
                trajectory_masks = [mask_t0] + method_interps[method_name] + [mask_t1]
                # Interpolated-only (no endpoints, for shape/path metrics)
                interp_masks = method_interps[method_name]

                # Compute per-step properties
                volumes = [mask_area(m) for m in trajectory_masks]
                perimeters = [mask_perimeter(m) for m in interp_masks]
                centroids = [mask_centroid(m) for m in interp_masks]
                centroids = np.array(centroids)

                # Metric 1: Volume monotonicity (uses full trajectory with endpoints)
                violation = volume_monotonicity_violation(volumes)
                results[method_name]['mono_violations'].append(violation)

                # Metric 2: Perimeter TV (interpolated frames only)
                ptv = perimeter_total_variation(perimeters)
                results[method_name]['perim_tv'].append(ptv)

                # Metric 3: Centroid curvature (interpolated frames only)
                curv = centroid_curvature(centroids)
                results[method_name]['centroid_curv'].append(curv)

                # Store for plotting (collect all valid trajectories for aggregation)
                if args.plot:
                    plot_data[method_name].append({
                        'alphas': [0.0] + list(args.alphas) + [1.0],
                        'volumes': volumes,
                    })

    print(f"\nTotal valid trajectory pairs: {n_valid_pairs}")
    print(f"Skipped (too small masks): {n_skipped}")

    # ─── Aggregate and report ───────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"Trajectory Smoothness Metrics")
    print(f"{'='*70}")

    header = f"{'Metric':<35} {'OT Mask':>12} {'Linear Mask':>12} {'Random Matching':>18}"
    print(header)
    print('-' * len(header))

    summary = {}
    for m in method_names:
        mono_rate = 100 * np.mean(results[m]['mono_violations'])
        ptv_arr = np.array(results[m]['perim_tv'])
        curv_arr = np.array(results[m]['centroid_curv'])

        summary[m] = {
            'mono_violation_pct': mono_rate,
            'perim_tv_median': float(np.median(ptv_arr)),
            'perim_tv_q25': float(np.percentile(ptv_arr, 25)),
            'perim_tv_q75': float(np.percentile(ptv_arr, 75)),
            'centroid_curv_median': float(np.median(curv_arr)),
            'centroid_curv_q25': float(np.percentile(curv_arr, 25)),
            'centroid_curv_q75': float(np.percentile(curv_arr, 75)),
            'n_trajectories': len(results[m]['mono_violations']),
        }

    # Print table
    header = f"{'Metric':<35} {'OT Mask':>22} {'Linear Mask':>22} {'Random Matching':>22}"
    print(header)
    print('-' * len(header))

    mono_str = {m: f"{summary[m]['mono_violation_pct']:.1f}%" for m in method_names}
    ptv_str = {m: (f"{summary[m]['perim_tv_median']:.2f} "
                    f"[{summary[m]['perim_tv_q25']:.2f}, {summary[m]['perim_tv_q75']:.2f}]")
               for m in method_names}
    curv_str = {m: (f"{summary[m]['centroid_curv_median']:.3f} "
                     f"[{summary[m]['centroid_curv_q25']:.3f}, {summary[m]['centroid_curv_q75']:.3f}]")
                for m in method_names}

    print(f"{'Non-monotonic volume (%) ↓':<35} "
          f"{mono_str['OT Mask']:>22} {mono_str['Linear Mask']:>22} {mono_str['Random Matching']:>22}")
    print(f"{'Perimeter TV (norm.) ↓':<35} "
          f"{ptv_str['OT Mask']:>22} {ptv_str['Linear Mask']:>22} {ptv_str['Random Matching']:>22}")
    print(f"{'CoM curvature (rad) ↓':<35} "
          f"{curv_str['OT Mask']:>22} {curv_str['Linear Mask']:>22} {curv_str['Random Matching']:>22}")

    # ─── Save CSV ───────────────────────────────────────────────────────────

    csv_path = output_dir / 'trajectory_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'OT Mask', 'Linear Mask', 'Random Matching'])
        writer.writerow(['Non-monotonic volume (%)',
                          f"{summary['OT Mask']['mono_violation_pct']:.1f}",
                          f"{summary['Linear Mask']['mono_violation_pct']:.1f}",
                          f"{summary['Random Matching']['mono_violation_pct']:.1f}"])
        def _csv_median_iqr(m, key):
            s = summary[m]
            return f"{s[f'{key}_median']:.3f} [{s[f'{key}_q25']:.3f}, {s[f'{key}_q75']:.3f}]"
        writer.writerow(['Perimeter TV (median[IQR])',
                          _csv_median_iqr('OT Mask', 'perim_tv'),
                          _csv_median_iqr('Linear Mask', 'perim_tv'),
                          _csv_median_iqr('Random Matching', 'perim_tv')])
        writer.writerow(['CoM curvature (median[IQR])',
                          _csv_median_iqr('OT Mask', 'centroid_curv'),
                          _csv_median_iqr('Linear Mask', 'centroid_curv'),
                          _csv_median_iqr('Random Matching', 'centroid_curv')])
    print(f"\nCSV saved to: {csv_path}")

    # ─── Save JSON ──────────────────────────────────────────────────────────

    json_path = output_dir / 'trajectory_metrics.json'
    json_out = {
        'n_trajectories': n_valid_pairs,
        'n_skipped': n_skipped,
        'alphas': args.alphas,
        'methods': summary,
        'raw': {m: {
            'mono_violations': [bool(v) for v in results[m]['mono_violations']],
            'perim_tv': [float(v) for v in results[m]['perim_tv']],
            'centroid_curv': [float(v) for v in results[m]['centroid_curv']],
        } for m in method_names},
    }
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"JSON saved to: {json_path}")

    # ─── LaTeX table fragment ───────────────────────────────────────────────

    def bold_best(vals, lower_better=True):
        """Return list of LaTeX strings with best value bolded."""
        best = min(vals) if lower_better else max(vals)
        out = []
        for v in vals:
            s = f"{v:.1f}" if isinstance(v, float) and v > 1 else f"{v:.2f}" if isinstance(v, float) and v > 0.01 else f"{v:.3f}"
            if abs(v - best) < 1e-6:
                s = r'\textbf{' + s + '}'
            out.append(s)
        return out

    latex_lines = []
    latex_lines.append(r'\begin{table}[t]')
    latex_lines.append(r'\caption{Trajectory smoothness metrics for mask interpolation methods. '
                       r'$N=' + str(n_valid_pairs) + r'$ consecutive timepoint pairs. '
                       r'Best in \textbf{bold}.}\label{tab:trajectory}')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\begin{tabular}{lccc}')
    latex_lines.append(r'\hline')
    latex_lines.append(r'Metric & OT Mask (Ours) & Linear Mask & Random Matching \\')
    latex_lines.append(r'\hline')

    # Row 1: Monotonicity violation
    vals = [summary[m]['mono_violation_pct'] for m in method_names]
    b = bold_best(vals, lower_better=True)
    latex_lines.append(f'Non-monotonic vol. (\\%)$\\downarrow$ & {b[0]} & {b[1]} & {b[2]} \\\\')

    # Row 2: Perimeter TV
    vals_median = [summary[m]['perim_tv_median'] for m in method_names]
    best_idx = np.argmin(vals_median)
    ptv_cells = []
    for i, m in enumerate(method_names):
        s = summary[m]
        cell = f"{s['perim_tv_median']:.2f} [{s['perim_tv_q25']:.2f}, {s['perim_tv_q75']:.2f}]"
        if i == best_idx:
            cell = (r'\textbf{' + f"{s['perim_tv_median']:.2f}" + r'}'
                    f" [{s['perim_tv_q25']:.2f}, {s['perim_tv_q75']:.2f}]")
        ptv_cells.append(cell)
    latex_lines.append(f'Perimeter TV$\\downarrow$ & {ptv_cells[0]} & {ptv_cells[1]} & {ptv_cells[2]} \\\\')

    # Row 3: CoM curvature
    vals_median = [summary[m]['centroid_curv_median'] for m in method_names]
    best_idx = np.argmin(vals_median)
    curv_cells = []
    for i, m in enumerate(method_names):
        s = summary[m]
        cell = f"{s['centroid_curv_median']:.3f} [{s['centroid_curv_q25']:.3f}, {s['centroid_curv_q75']:.3f}]"
        if i == best_idx:
            cell = (r'\textbf{' + f"{s['centroid_curv_median']:.3f}" + r'}'
                    f" [{s['centroid_curv_q25']:.3f}, {s['centroid_curv_q75']:.3f}]")
        curv_cells.append(cell)
    latex_lines.append(f'CoM curvature (rad)$\\downarrow$ & {curv_cells[0]} & {curv_cells[1]} & {curv_cells[2]} \\\\')

    latex_lines.append(r'\hline')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')

    latex_str = '\n'.join(latex_lines)
    tex_path = output_dir / 'table_trajectory.tex'
    tex_path.write_text(latex_str)
    print(f"LaTeX table saved to: {tex_path}")
    print()
    print(latex_str)

    # ─── Save plot data for standalone replotting ──────────────────────────

    if args.plot and any(len(v) > 0 for v in plot_data.values()):
        plot_json_path = output_dir / 'plot_data.json'
        # Convert plot_data to serializable format
        plot_data_serializable = {}
        for m in method_names:
            plot_data_serializable[m] = [
                {'alphas': traj['alphas'], 'volumes': traj['volumes']}
                for traj in plot_data[m]
            ]
        with open(plot_json_path, 'w') as f:
            json.dump(plot_data_serializable, f)
        print(f"Plot data saved to: {plot_json_path}")

    # ─── Optional volume-vs-alpha plot ──────────────────────────────────────

    if args.plot and any(len(v) > 0 for v in plot_data.values()):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = {'OT Mask': '#2196F3', 'Linear Mask': '#FF9800', 'Random Matching': '#9E9E9E'}
            linestyles = {'OT Mask': '-', 'Linear Mask': '--', 'Random Matching': ':'}

            for method_name in method_names:
                # Normalize all trajectories and stack into array
                all_norm_trajs = []
                for traj in plot_data[method_name]:
                    vols = traj['volumes']
                    v0, v1 = vols[0], vols[-1]
                    if abs(v1 - v0) > 1:
                        norm_vols = [(v - v0) / (v1 - v0) for v in vols]
                    else:
                        norm_vols = [0.5] * len(vols)
                    all_norm_trajs.append(norm_vols)

                if not all_norm_trajs:
                    continue
                arr = np.array(all_norm_trajs)  # (N_traj, N_alpha_points)
                alphas = plot_data[method_name][0]['alphas']
                mean_vals = arr.mean(axis=0)
                std_vals = arr.std(axis=0)

                ax.plot(alphas, mean_vals, color=colors[method_name],
                        linestyle=linestyles[method_name],
                        linewidth=2, label=method_name)

            # Reference: ideal linear
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8, label='Ideal linear')
            ax.set_xlabel(r'Interpolation $\alpha$')
            ax.set_ylabel('Normalized volume')
            ax.set_title('Volume trajectories')
            ax.legend(fontsize=8)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.15, 1.15)
            plt.tight_layout()

            plot_path = output_dir / 'volume_vs_alpha.pdf'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / 'volume_vs_alpha.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Volume plot saved to: {plot_path}")
        except ImportError as e:
            print(f"Skipping plot (missing dependency: {e})")


if __name__ == '__main__':
    main()
