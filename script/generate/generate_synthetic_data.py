"""
Batch generation of synthetic CT data using OT-interpolated masks + flow matching.

For each (patient, slice_position) with >=2 tumor-bearing timepoints:
1. Compute OT-interpolated masks between consecutive tumor timepoint pairs
2. Use the mask-conditioned flow-matching model to synthesize CT images
3. Save the synthetic timepoints as a tensor file

Usage:
    python script/generate_synthetic_data.py \
        --data_path data_processed/longitudinal_data.pt \
        --ckpt_path path/to/controlnet_fm.pth \
        --output_dir data_processed/synthetic \
        --interp_steps 0.25 0.5 0.75

    # Fewer interpolation points for faster generation
    python script/generate_synthetic_data.py \
        --data_path data_processed/longitudinal_data.pt \
        --ckpt_path path/to/controlnet_fm.pth \
        --output_dir data_processed/synthetic \
        --interp_steps 0.5
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from src.models.controlnet import create_controlnet_inpaint
from src.models.flow_matching import FlowMatchingScheduler
from src.data.dataset_slices import LongitudinalCTDataset
from src.inference.sample_controlnet_fm import sample_fm
from src.ot.interpolation import compute_ot_interpolated_mask, compute_random_interpolated_mask


def compute_linear_interpolated_mask(mask_t0, mask_t1, t=0.5):
    """
    Naive linear interpolation of masks in pixel space (no OT).
    Threshold at 0.5 to get binary mask.
    """
    mask_float = (1 - t) * mask_t0.float() + t * mask_t1.float()
    return (mask_float > 0.5).float()


def compute_interpolation_tasks(item, interp_steps, interp_method='ot'):
    """
    For a single (patient, slice) entry, compute all OT-interpolated masks.

    Args:
        item: dict from LongitudinalCTDataset (eval mode)
        interp_steps: list of t values for interpolation (e.g. [0.25, 0.5, 0.75])

    Returns:
        list of dicts with keys: mask, bg_ct, interp_t, pair
    """
    ct_series = item['ct_series']     # (T, 1, H, W)
    mask_series = item['mask_series']  # (T, 1, H, W)

    # Identify tumor-bearing timepoints (for background selection)
    tumor_tp_indices = []
    for t_idx in range(ct_series.shape[0]):
        if mask_series[t_idx].sum() >= 3:  # at least 3 pixels
            tumor_tp_indices.append(t_idx)

    if len(tumor_tp_indices) < 2:
        return []

    # Select background: timepoint with smallest tumor
    tumor_sizes = [(i, mask_series[i].sum().item()) for i in tumor_tp_indices]
    bg_t_idx = min(tumor_sizes, key=lambda x: x[1])[0]
    bg_ct = ct_series[bg_t_idx]  # (1, H, W)

    tumor_tp_set = set(tumor_tp_indices)
    tasks = []
    # Form pairs from EVERY consecutive timepoint; only interpolate if both have tumor
    for ti in range(ct_series.shape[0] - 1):
        tj = ti + 1
        if ti not in tumor_tp_set or tj not in tumor_tp_set:
            continue

        mask_ti = mask_series[ti]  # (1, H, W)
        mask_tj = mask_series[tj]  # (1, H, W)

        for t_val in interp_steps:
            try:
                if interp_method == 'ot':
                    mask_interp = compute_ot_interpolated_mask(mask_ti, mask_tj, t=t_val)
                elif interp_method == 'random':
                    mask_interp = compute_random_interpolated_mask(mask_ti, mask_tj, t=t_val)
                else:
                    mask_interp = compute_linear_interpolated_mask(mask_ti, mask_tj, t=t_val)
            except Exception as e:
                print(f"  Warning: {interp_method} interpolation failed for pair ({ti},{tj}) t={t_val}: {e}")
                continue

            # Skip if interpolated mask is empty
            if mask_interp.sum() < 3:
                continue

            tasks.append({
                'mask': mask_interp,
                'bg_ct': bg_ct,
                'interp_t': t_val,
                'pair': (ti, tj),
            })

    return tasks


def generate_for_patient(patient_id, dataset, model, scheduler, device, args):
    """
    Generate all synthetic CTs for one patient.

    Returns:
        list of dicts with synthetic data
    """
    slices = dataset.get_patient_slices(patient_id)
    results = []

    for s in slices:
        # Only process slices with >=2 tumor timepoints
        if len(s['timepoints']) < 2:
            continue

        tasks = compute_interpolation_tasks(s, args.interp_steps, args.interp_method)
        if not tasks:
            continue

        # Batch synthesis: group masks and backgrounds
        masks_list = [t['mask'] for t in tasks]
        bgs_list = [t['bg_ct'] for t in tasks]

        masks_batch = torch.stack(masks_list)  # (N, 1, H, W)
        bgs_batch = torch.stack(bgs_list)      # (N, 1, H, W)

        # Process in mini-batches
        for start in range(0, len(tasks), args.gen_batch_size):
            end = min(start + args.gen_batch_size, len(tasks))
            masks_mb = masks_batch[start:end].to(device)
            bgs_mb = bgs_batch[start:end].to(device)

            synth_cts = sample_fm(
                model, scheduler, bgs_mb, masks_mb,
                steps=args.steps, method=args.method,
                controlnet_scale=args.controlnet_scale,
            )

            for i in range(synth_cts.shape[0]):
                task_idx = start + i
                results.append({
                    'ct': synth_cts[i].cpu(),              # (1, H, W)
                    'mask': tasks[task_idx]['mask'].cpu(),  # (1, H, W)
                    'patient_id': patient_id,
                    'slice_idx': s['slice_idx'],
                    'interp_t': tasks[task_idx]['interp_t'],
                    'pair': tasks[task_idx]['pair'],
                })

    return results


def load_manifest(output_dir):
    """Load set of completed patient IDs from manifest."""
    manifest_path = output_dir / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            return set(json.load(f)['completed_patients'])
    return set()


def save_manifest(output_dir, completed_patients):
    """Save manifest of completed patient IDs."""
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({'completed_patients': sorted(completed_patients)}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Batch generate synthetic CT data')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to longitudinal_data.pt')

    # Model
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to flow matching checkpoint')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA weights if available')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')

    # Sampling
    parser.add_argument('--steps', type=int, default=50,
                        help='ODE solver steps')
    parser.add_argument('--method', type=str, default='heun',
                        choices=['euler', 'heun'])
    parser.add_argument('--controlnet_scale', type=float, default=1.5)
    parser.add_argument('--t_eps', type=float, default=0.05)

    # Interpolation
    parser.add_argument('--interp_steps', type=float, nargs='+',
                        default=[0.25, 0.5, 0.75],
                        help='Interpolation t values (e.g. 0.25 0.5 0.75)')
    parser.add_argument('--interp_method', type=str, default='ot',
                        choices=['ot', 'linear', 'random'],
                        help='Mask interpolation method: ot (optimal transport), '
                             'linear (naive pixel), or random (random point cloud matching)')

    # Generation
    parser.add_argument('--gen_batch_size', type=int, default=8,
                        help='Batch size for flow matching inference')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data_processed/synthetic_ot_mask for ot, '
                             'data_processed/synthetic_linear_mask for linear)')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Set default output_dir based on interp_method
    if args.output_dir is None:
        args.output_dir = f'data_processed/synthetic_{args.interp_method}_mask'

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Interpolation method: {args.interp_method}")
    print(f"Interpolation steps: {args.interp_steps}")

    # Load dataset in eval mode (returns all timepoints per entry)
    print(f"\nLoading dataset from {args.data_path}")
    dataset = LongitudinalCTDataset(args.data_path, mode='eval', filter_no_tumor=True)
    patients = dataset.get_available_patients()
    print(f"Total patients with tumor: {len(patients)}")

    # Load model
    print(f"\nLoading model from {args.ckpt_path}")
    model = create_controlnet_inpaint(image_size=args.image_size).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)

    if args.use_ema and 'ema_state_dict' in ckpt:
        print("Loading EMA weights")
        model.load_state_dict(ckpt['ema_state_dict'])
    else:
        model.unet.load_state_dict(ckpt['unet_state_dict'])
        model.controlnet.load_state_dict(ckpt['controlnet_state_dict'])
    model.eval()

    scheduler = FlowMatchingScheduler(t_eps=args.t_eps)

    # Check for completed patients (resume support)
    completed = load_manifest(output_dir)
    remaining = [p for p in patients if p not in completed]
    if completed:
        print(f"Resuming: {len(completed)} patients already done, {len(remaining)} remaining")

    # Load previously saved data for incremental append
    save_path = output_dir / 'synthetic_data.pt'
    if save_path.exists() and completed:
        all_synthetic = torch.load(save_path)
        print(f"Loaded {len(all_synthetic)} existing synthetic slices")
    else:
        all_synthetic = []

    # Generate synthetic data for each patient
    start_time = time.time()

    for pid_idx, pid in enumerate(tqdm(remaining, desc='Patients')):
        print(f"\n[{pid_idx+1}/{len(remaining)}] Patient {pid}")

        patient_results = generate_for_patient(pid, dataset, model, scheduler, device, args)

        if patient_results:
            all_synthetic.extend(patient_results)
            print(f"  Generated {len(patient_results)} synthetic slices")
        else:
            print(f"  No eligible slices (need >=2 tumor timepoints)")

        # Save data and manifest together after each patient (crash-safe)
        torch.save(all_synthetic, save_path)
        completed.add(pid)
        save_manifest(output_dir, completed)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Generation complete in {elapsed/60:.1f} minutes")
    print(f"Total synthetic slices: {len(all_synthetic)}")

    if all_synthetic:
        # Print statistics
        patient_counts = {}
        for d in all_synthetic:
            pid = d['patient_id']
            patient_counts[pid] = patient_counts.get(pid, 0) + 1

        counts = list(patient_counts.values())
        print(f"Patients with synthetic data: {len(patient_counts)}")
        print(f"Synthetic slices per patient: min={min(counts)}, max={max(counts)}, "
              f"mean={sum(counts)/len(counts):.1f}")
    else:
        print("No synthetic data generated.")

    # Save generation config
    config_path = output_dir / 'generation_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")


if __name__ == '__main__':
    main()
