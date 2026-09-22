"""
Baseline: naive pixel-space linear interpolation between consecutive timepoints.

No generative model is used. For each consecutive tumor-bearing pair (ti, tj),
the interpolated CT and mask are computed as simple weighted averages:
    ct_interp  = (1-t) * ct_ti  + t * ct_tj
    mask_interp = threshold((1-t) * mask_ti + t * mask_tj, 0.5)

Output format matches the synthetic_data.pt produced by generate_synthetic_data.py.py.

Usage:
    python script/generate/generate_pixel_interp.py \
        --data_path data_processed/longitudinal_data.pt \
        --output_dir data_processed/synthetic_pixel_interp \
        --interp_steps 0.25 0.5 0.75
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from src.data.dataset_slices import LongitudinalCTDataset


def main():
    parser = argparse.ArgumentParser(
        description='Generate pixel-interpolated baseline data')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to longitudinal_data.pt')
    parser.add_argument('--output_dir', type=str,
                        default='data_processed/synthetic_pixel_interp')
    parser.add_argument('--interp_steps', type=float, nargs='+',
                        default=[0.25, 0.5, 0.75])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.data_path}")
    dataset = LongitudinalCTDataset(args.data_path, mode='eval', filter_no_tumor=True)
    patients = dataset.get_available_patients()
    print(f"Total patients with tumor: {len(patients)}")
    print(f"Interpolation steps: {args.interp_steps}")

    all_synthetic = []

    for pid in tqdm(patients, desc='Patients'):
        slices = dataset.get_patient_slices(pid)

        for s in slices:
            ct_series = s['ct_series']      # (T, 1, H, W)
            mask_series = s['mask_series']   # (T, 1, H, W)
            T = ct_series.shape[0]

            # Identify tumor-bearing timepoints
            tumor_tps = set()
            for t_idx in range(T):
                if mask_series[t_idx].sum() >= 3:
                    tumor_tps.add(t_idx)

            # Interpolate between consecutive tumor-bearing pairs
            for ti in range(T - 1):
                tj = ti + 1
                if ti not in tumor_tps or tj not in tumor_tps:
                    continue

                ct_ti = ct_series[ti].float()    # (1, H, W)
                ct_tj = ct_series[tj].float()
                mask_ti = mask_series[ti].float()
                mask_tj = mask_series[tj].float()

                for t_val in args.interp_steps:
                    ct_interp = (1 - t_val) * ct_ti + t_val * ct_tj
                    mask_interp = ((1 - t_val) * mask_ti + t_val * mask_tj > 0.5).float()

                    if mask_interp.sum() < 3:
                        continue

                    all_synthetic.append({
                        'ct': ct_interp,           # (1, H, W)
                        'mask': mask_interp,       # (1, H, W)
                        'patient_id': pid,
                        'slice_idx': s['slice_idx'],
                        'interp_t': t_val,
                        'pair': (ti, tj),
                    })

    save_path = output_dir / 'synthetic_data.pt'
    torch.save(all_synthetic, save_path)
    print(f"\nSaved {len(all_synthetic)} pixel-interpolated slices to {save_path}")

    # Stats
    patient_counts = {}
    for d in all_synthetic:
        pid = d['patient_id']
        patient_counts[pid] = patient_counts.get(pid, 0) + 1

    if patient_counts:
        counts = list(patient_counts.values())
        print(f"Patients with data: {len(patient_counts)}")
        print(f"Slices per patient: min={min(counts)}, max={max(counts)}, "
              f"mean={sum(counts)/len(counts):.1f}")

    config_path = output_dir / 'generation_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    main()
