"""
Compare SSIM / PSNR between synthetic CTs and real CTs across generation methods.

For each synthetic sample, match it back to the real endpoint CTs using
(patient_id, slice_idx, pair) and compute:
  - Background SSIM: non-tumor region
  - Tumor SSIM: tumor region only (captures texture quality)
  - Full SSIM: entire image
  - Tumor Std: HU std within tumor (lower = more uniform texture)

Usage:
    python script/eval/compare_ssim.py

    # Custom synthetic dirs
    python script/eval/compare_ssim.py \
        --methods \
            ot_fm   data_processed/synthetic \
            random  data_processed/synthetic_random_mask

    # Only tumor-region SSIM
    python script/eval/compare_ssim.py --region tumor

    # Output JSON for downstream analysis
    python script/eval/compare_ssim.py --output outputs/comparison/ssim_comparison.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset_slices import LongitudinalCTDataset


DEFAULT_METHODS = [
    ('OT + FM (Ours)', 'data_processed/synthetic'),
    ('Random + FM',    'data_processed/synthetic_random_mask'),
    ('Linear + FM',    'data_processed/synthetic_linear_mask'),
]


def build_real_lookup(dataset):
    """Build lookup: (patient_id, slice_idx) → dict with ct_series, mask_series, timepoints."""
    lookup = {}
    for i in range(len(dataset)):
        item = dataset[i]
        key = (item['patient_id'], item['slice_idx'])
        lookup[key] = item
    return lookup


def get_real_endpoint_ct(real_item, pair, interp_t):
    """Get the real CT closest to this synthetic sample's interpolation position.

    For interp_t <= 0.5, use the first endpoint of the pair.
    For interp_t > 0.5, use the second endpoint.
    Returns (real_ct, real_mask) as numpy arrays, or (None, None) if not found.
    """
    ti, tj = pair
    ct_series = real_item['ct_series']   # (T, 1, H, W)
    mask_series = real_item['mask_series']  # (T, 1, H, W)

    if ti >= ct_series.shape[0] or tj >= ct_series.shape[0]:
        return None, None

    # Choose closer endpoint
    ref_idx = ti if interp_t <= 0.5 else tj
    real_ct = ct_series[ref_idx, 0].numpy()      # (H, W)
    real_mask = mask_series[ref_idx, 0].numpy()   # (H, W)
    return real_ct, real_mask


def compute_metrics(synth_ct, real_ct, mask, region='all'):
    """Compute SSIM, PSNR, and texture std between synthetic and real CT.

    Args:
        synth_ct: (H, W) synthetic CT, float
        real_ct: (H, W) real CT, float
        mask: (H, W) binary tumor mask (for the synthetic sample's condition)
        region: 'all', 'tumor', or 'background'

    Returns:
        dict with ssim, psnr, tumor_std_synth, tumor_std_real
    """
    # Determine data range for SSIM/PSNR
    combined = np.concatenate([synth_ct.ravel(), real_ct.ravel()])
    data_range = combined.max() - combined.min()
    if data_range < 1e-8:
        data_range = 1.0

    mask_bool = mask > 0.5

    results = {}

    if region in ('all', 'full'):
        results['ssim_full'] = ssim(real_ct, synth_ct, data_range=data_range)
        results['psnr_full'] = psnr(real_ct, synth_ct, data_range=data_range)

    if region in ('all', 'tumor'):
        if mask_bool.sum() >= 10:
            # Crop to tumor bounding box + margin for SSIM window
            ys, xs = np.where(mask_bool)
            margin = 7  # SSIM default window is 7
            y0 = max(0, ys.min() - margin)
            y1 = min(mask_bool.shape[0], ys.max() + margin + 1)
            x0 = max(0, xs.min() - margin)
            x1 = min(mask_bool.shape[1], xs.max() + margin + 1)

            crop_synth = synth_ct[y0:y1, x0:x1]
            crop_real = real_ct[y0:y1, x0:x1]

            # Only compute if crop is large enough for SSIM window
            if crop_synth.shape[0] >= 7 and crop_synth.shape[1] >= 7:
                results['ssim_tumor'] = ssim(crop_real, crop_synth, data_range=data_range)
                results['psnr_tumor'] = psnr(crop_real, crop_synth, data_range=data_range)
            else:
                results['ssim_tumor'] = float('nan')
                results['psnr_tumor'] = float('nan')

            # Texture uniformity: std of HU within tumor
            results['tumor_std_synth'] = float(synth_ct[mask_bool].std())
            results['tumor_std_real'] = float(real_ct[mask_bool].std())
        else:
            results['ssim_tumor'] = float('nan')
            results['psnr_tumor'] = float('nan')
            results['tumor_std_synth'] = float('nan')
            results['tumor_std_real'] = float('nan')

    if region in ('all', 'background'):
        bg_mask = ~mask_bool
        if bg_mask.sum() >= 100:
            # For background, use full image but mask out tumor
            bg_synth = synth_ct.copy()
            bg_real = real_ct.copy()
            # Set tumor region to same value so it doesn't affect SSIM
            fill_val = real_ct[bg_mask].mean()
            bg_synth[mask_bool] = fill_val
            bg_real[mask_bool] = fill_val
            results['ssim_bg'] = ssim(bg_real, bg_synth, data_range=data_range)
        else:
            results['ssim_bg'] = float('nan')

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare SSIM/PSNR across synthetic methods')
    parser.add_argument('--data_path', type=str,
                        default='data_processed/longitudinal_data.pt',
                        help='Path to real data')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Method specs: name1 path1 name2 path2 ... '
                             '(default: OT FM, Random FM, Linear FM)')
    parser.add_argument('--region', type=str, default='all',
                        choices=['all', 'tumor', 'background', 'full'],
                        help='Which region to compute SSIM for')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max samples per method (0 = all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    # Parse methods
    if args.methods:
        if len(args.methods) % 2 != 0:
            raise ValueError("--methods must be pairs: name1 path1 name2 path2 ...")
        methods = [(args.methods[i], args.methods[i+1])
                   for i in range(0, len(args.methods), 2)]
    else:
        methods = [(name, path) for name, path in DEFAULT_METHODS
                   if Path(path).exists()]

    if not methods:
        print("No synthetic data directories found.")
        return

    print(f"Methods to compare: {[m[0] for m in methods]}")
    print(f"Region: {args.region}")

    # Load real data
    print(f"\nLoading real data from {args.data_path}")
    dataset = LongitudinalCTDataset(args.data_path, mode='eval', filter_no_tumor=True)
    real_lookup = build_real_lookup(dataset)
    print(f"  {len(real_lookup)} real (patient, slice) entries")

    # Process each method
    all_results = {}

    for method_name, synth_path in methods:
        synth_file = Path(synth_path) / 'synthetic_data.pt'
        if not synth_file.exists():
            print(f"\n[{method_name}] {synth_file} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[{method_name}] Loading {synth_file}")
        synth_data = torch.load(synth_file, map_location='cpu', weights_only=False)
        print(f"  {len(synth_data)} synthetic samples")

        if args.max_samples > 0:
            synth_data = synth_data[:args.max_samples]

        metrics_list = []
        skipped = 0

        for item in tqdm(synth_data, desc=f"  {method_name}"):
            pid = item['patient_id']
            sidx = item['slice_idx']
            pair = tuple(item['pair'])
            interp_t = item['interp_t']

            # Look up real data
            key = (pid, sidx)
            if key not in real_lookup:
                skipped += 1
                continue

            real_item = real_lookup[key]
            real_ct, real_mask = get_real_endpoint_ct(real_item, pair, interp_t)
            if real_ct is None:
                skipped += 1
                continue

            # Get synthetic CT
            synth_ct = item['ct']
            if isinstance(synth_ct, torch.Tensor):
                synth_ct = synth_ct.numpy()
            if synth_ct.ndim == 3:
                synth_ct = synth_ct[0]  # (1, H, W) → (H, W)

            # Use the synthetic sample's condition mask for region selection
            synth_mask = item['mask']
            if isinstance(synth_mask, torch.Tensor):
                synth_mask = synth_mask.numpy()
            if synth_mask.ndim == 3:
                synth_mask = synth_mask[0]

            m = compute_metrics(synth_ct, real_ct, synth_mask, region=args.region)
            m['patient_id'] = int(pid)
            m['slice_idx'] = int(sidx)
            m['interp_t'] = float(interp_t)
            metrics_list.append(m)

        if skipped:
            print(f"  Skipped {skipped} samples (no matching real data)")

        if not metrics_list:
            print(f"  No valid samples!")
            continue

        # Aggregate
        metric_keys = [k for k in metrics_list[0] if k not in ('patient_id', 'slice_idx', 'interp_t')]
        summary = {}
        for k in metric_keys:
            vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
            if vals:
                summary[k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'median': float(np.median(vals)),
                    'n': len(vals),
                }

        all_results[method_name] = {
            'summary': summary,
            'n_samples': len(metrics_list),
            'path': str(synth_path),
        }

        # Per interp_t breakdown
        by_t = defaultdict(list)
        for m in metrics_list:
            by_t[m['interp_t']].append(m)

        per_t = {}
        for t_val in sorted(by_t.keys()):
            items = by_t[t_val]
            t_summary = {}
            for k in metric_keys:
                vals = [m[k] for m in items if not np.isnan(m[k])]
                if vals:
                    t_summary[k] = {'mean': float(np.mean(vals)), 'n': len(vals)}
            per_t[str(t_val)] = t_summary
        all_results[method_name]['per_interp_t'] = per_t

    # Print comparison table
    print(f"\n{'='*80}")
    print("SSIM / PSNR Comparison")
    print(f"{'='*80}")

    # Collect all metric keys across methods
    all_metric_keys = set()
    for r in all_results.values():
        all_metric_keys.update(r['summary'].keys())
    all_metric_keys = sorted(all_metric_keys)

    # Header
    name_width = max(len(name) for name in all_results) + 2
    header = f"{'Method':<{name_width}}"
    for k in all_metric_keys:
        header += f"  {k:>18}"
    print(header)
    print("-" * len(header))

    # Rows
    for method_name, res in all_results.items():
        row = f"{method_name:<{name_width}}"
        for k in all_metric_keys:
            if k in res['summary']:
                s = res['summary'][k]
                row += f"  {s['mean']:>8.4f}±{s['std']:<7.4f}"
            else:
                row += f"  {'--':>18}"
        print(row)

    # Per interp_t detail
    print(f"\n{'='*80}")
    print("Per interp_t breakdown (mean values)")
    print(f"{'='*80}")

    for method_name, res in all_results.items():
        if 'per_interp_t' not in res:
            continue
        print(f"\n  [{method_name}]")
        for t_val, t_summary in sorted(res['per_interp_t'].items()):
            parts = [f"    t={t_val}:"]
            for k, v in sorted(t_summary.items()):
                parts.append(f"{k}={v['mean']:.4f}(n={v['n']})")
            print("  ".join(parts))

    # Save JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
