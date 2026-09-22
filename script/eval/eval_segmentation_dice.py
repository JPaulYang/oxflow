"""
Evaluate synthetic CT quality by running nnUNet segmentation and computing Dice score.

For each synthetic CT slice:
1. Run nnUNet inference to predict tumor segmentation
2. Compare with the OT-interpolated conditioning mask using Dice score

Reports overall, per-patient, and per-interpolation-step statistics.

Prerequisite:
    - Trained nnUNet model (see script/train/prepare_nnunet_data.py)
    - Generated synthetic data (data_processed/synthetic/synthetic_data.pt)

Usage:
    export nnUNet_results="data_processed/nnUNet_results"

    python script/eval/eval_segmentation_dice.py \
        --synthetic_path data_processed/synthetic/synthetic_data.pt \
        --dataset_id 1 \
        --config 2d \
        --fold 0

    # Also evaluate on real data (as upper-bound reference)
    python script/eval/eval_segmentation_dice.py \
        --synthetic_path data_processed/synthetic/synthetic_data.pt \
        --real_path data_processed/longitudinal_data.pt \
        --dataset_id 1 \
        --config 2d \
        --fold 0
"""

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm


def dice_score(pred, target, smooth=1e-5):
    """Compute Dice score between two binary masks."""
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    intersection = (pred & target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def load_nnunet_predictor(dataset_id, config, fold, device):
    """Load nnUNet predictor for inference."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    nnunet_results = os.environ.get('nnUNet_results')
    if nnunet_results is None:
        raise ValueError("nnUNet_results environment variable not set")

    # Find the model directory
    results_dir = Path(nnunet_results)
    dataset_name = None
    for d in results_dir.iterdir():
        if d.is_dir() and d.name.startswith(f"Dataset{dataset_id:03d}_"):
            dataset_name = d.name
            break

    if dataset_name is None:
        raise FileNotFoundError(
            f"No trained model found for Dataset{dataset_id:03d} in {results_dir}")

    model_folder = results_dir / dataset_name / f"nnUNetTrainer__nnUNetPlans__{config}"

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_mirroring=False,  # faster inference without test-time augmentation
        device=torch.device(device),
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(fold,),
    )
    print(f"Loaded nnUNet model from {model_folder}")
    return predictor


def predict_batch_slices(predictor, ct_tensors):
    """
    Run nnUNet prediction on a batch of 2D CT slices.

    Writes all slices to a single temp directory, calls predict_from_files
    once, and reads all predictions back. This lets nnUNet pipeline
    preprocessing and GPU inference across the batch.

    Args:
        predictor: nnUNetPredictor
        ct_tensors: list of (1, H, W) tensors

    Returns:
        list of (H, W) numpy arrays, binary predicted masks
    """
    if not ct_tensors:
        return []

    predictions = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        out_dir = tmpdir / "output"
        out_dir.mkdir()

        # Write all inputs at once
        input_files = []
        for j, ct in enumerate(ct_tensors):
            ct_np = ct.squeeze().numpy().astype(np.float32)
            arr_3d = ct_np[:, :, np.newaxis]  # (H, W, 1)
            fname = f"case_{j:05d}_0000.nii.gz"
            nib.save(nib.Nifti1Image(arr_3d, np.eye(4)), str(tmpdir / fname))
            input_files.append([str(tmpdir / fname)])

        # Single batched predict call
        predictor.predict_from_files(
            input_files,
            str(out_dir),
            save_probabilities=False,
        )

        # Read all outputs
        for j in range(len(ct_tensors)):
            pred_path = out_dir / f"case_{j:05d}.nii.gz"
            pred_mask = nib.load(str(pred_path)).get_fdata()[:, :, 0]
            predictions.append((pred_mask > 0.5).astype(np.float32))

    return predictions


def save_checkpoint(results, checkpoint_path, last_data_idx, label=""):
    """Save intermediate results to avoid losing progress."""
    ckpt = {
        'last_data_idx': last_data_idx,
        'n_completed': len(results),
        'results': results,
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(ckpt, f)


def load_checkpoint(checkpoint_path):
    """Load checkpoint if it exists, return (results, last_data_idx)."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt['results']
        last_data_idx = ckpt.get('last_data_idx', ckpt['n_completed'])
        print(f"  Resuming from checkpoint: {len(results)} results, skipping data[:{last_data_idx}]")
        return results, last_data_idx
    return [], 0


def evaluate_synthetic(predictor, synthetic_path, device, checkpoint_dir=None, batch_size=64):
    """Evaluate Dice on synthetic data."""
    print(f"\nLoading synthetic data from {synthetic_path}")
    synthetic_data = torch.load(synthetic_path)
    print(f"Total synthetic samples: {len(synthetic_data)}")

    # Resume from checkpoint if available
    ckpt_path = Path(checkpoint_dir) / 'checkpoint_synthetic.json' if checkpoint_dir else None
    results, start_idx = load_checkpoint(ckpt_path) if ckpt_path else ([], 0)

    # Collect valid items (skip already-processed and empty masks)
    valid_items = []
    for i, item in enumerate(synthetic_data):
        if i < start_idx:
            continue
        mask_gt_np = (item['mask'].squeeze().numpy() > 0).astype(np.float32)
        if mask_gt_np.sum() < 3:
            continue
        valid_items.append((i, item, mask_gt_np))

    print(f"Remaining valid samples: {len(valid_items)}")

    # Process in batches
    pbar = tqdm(total=len(valid_items), desc="Evaluating synthetic")
    for b_start in range(0, len(valid_items), batch_size):
        batch = valid_items[b_start:b_start + batch_size]
        ct_tensors = [item['ct'] for _, item, _ in batch]

        pred_masks = predict_batch_slices(predictor, ct_tensors)

        for (orig_idx, item, mask_gt_np), pred_mask in zip(batch, pred_masks):
            d = dice_score(pred_mask, mask_gt_np)
            results.append({
                'dice': d,
                'patient_id': item['patient_id'],
                'slice_idx': item['slice_idx'],
                'interp_t': item['interp_t'],
                'pair': item['pair'],
            })

        pbar.update(len(batch))

        if ckpt_path:
            last_idx = batch[-1][0] + 1
            save_checkpoint(results, ckpt_path, last_idx, "synthetic")

    pbar.close()
    return results


def evaluate_real(predictor, real_path, device, checkpoint_dir=None, batch_size=64):
    """Evaluate Dice on real data (upper-bound reference)."""
    print(f"\nLoading real data from {real_path}")
    all_data = torch.load(real_path)

    # Flatten into evaluation list (already filtered for non-empty masks)
    eval_list = []
    for item in all_data:
        ct_slices = item['ct_slices']   # (T, 1, H, W)
        masks = item['masks']           # (T, 1, H, W)
        T = ct_slices.shape[0]
        for t in range(T):
            mask_gt = (masks[t, 0] > 0).float().numpy()
            if mask_gt.sum() < 3:
                continue
            eval_list.append((item, t, mask_gt))

    print(f"Total real tumor-bearing slices: {len(eval_list)}")

    # Resume from checkpoint if available
    ckpt_path = Path(checkpoint_dir) / 'checkpoint_real.json' if checkpoint_dir else None
    results, start_idx = load_checkpoint(ckpt_path) if ckpt_path else ([], 0)

    remaining = eval_list[start_idx:]
    print(f"Remaining slices to evaluate: {len(remaining)}")

    pbar = tqdm(total=len(remaining), desc="Evaluating real")
    for b_start in range(0, len(remaining), batch_size):
        batch = remaining[b_start:b_start + batch_size]
        ct_tensors = [item['ct_slices'][t] for item, t, _ in batch]

        pred_masks = predict_batch_slices(predictor, ct_tensors)

        for (item, t, mask_gt), pred_mask in zip(batch, pred_masks):
            d = dice_score(pred_mask, mask_gt)
            results.append({
                'dice': d,
                'patient_id': item['patient_id'],
                'slice_idx': item['slice_idx'],
                'timepoint': t,
            })

        pbar.update(len(batch))

        if ckpt_path:
            save_checkpoint(results, ckpt_path,
                            start_idx + b_start + len(batch), "real")

    pbar.close()
    return results


def print_summary(results, label):
    """Print summary statistics."""
    if not results:
        print(f"\n{label}: No results")
        return

    dices = np.array([r['dice'] for r in results])
    median = np.median(dices)
    q25, q75 = np.percentile(dices, [25, 75])
    print(f"\n{'='*60}")
    print(f"{label}: {len(dices)} samples")
    print(f"  Dice (median [IQR]): {median:.4f} [{q25:.4f}, {q75:.4f}]")
    print(f"  Dice range: [{np.min(dices):.4f}, {np.max(dices):.4f}]")

    # Per-patient
    patient_dices = defaultdict(list)
    for r in results:
        patient_dices[r['patient_id']].append(r['dice'])
    patient_medians = {pid: np.median(ds) for pid, ds in patient_dices.items()}
    vals = list(patient_medians.values())
    print(f"\n  Per-patient Dice ({len(patient_medians)} patients):")
    print(f"    Median of medians: {np.median(vals):.4f}")
    print(f"    Worst patient: {min(vals):.4f}")
    print(f"    Best patient: {max(vals):.4f}")

    # Per interp_t (synthetic only)
    if 'interp_t' in results[0]:
        interp_dices = defaultdict(list)
        for r in results:
            interp_dices[r['interp_t']].append(r['dice'])
        print(f"\n  Per interp_t:")
        for t_val in sorted(interp_dices.keys()):
            ds = np.array(interp_dices[t_val])
            t_med = np.median(ds)
            t_q25, t_q75 = np.percentile(ds, [25, 75])
            print(f"    t={t_val:.2f}: {t_med:.4f} [{t_q25:.4f}, {t_q75:.4f}] (n={len(ds)})")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate synthetic CT quality via nnUNet segmentation + Dice')
    parser.add_argument('--synthetic_path', type=str, required=True,
                        help='Path to synthetic_data.pt')
    parser.add_argument('--real_path', type=str, default=None,
                        help='Path to longitudinal_data.pt (for real data baseline)')
    parser.add_argument('--dataset_id', type=int, default=1,
                        help='nnUNet dataset ID')
    parser.add_argument('--config', type=str, default='2d',
                        help='nnUNet config (2d, 3d_fullres, etc.)')
    parser.add_argument('--fold', type=int, default=0,
                        help='nnUNet fold to use')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='outputs/eval_dice',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of slices per nnUNet inference batch')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load nnUNet predictor
    predictor = load_nnunet_predictor(
        args.dataset_id, args.config, args.fold, args.device)

    # Evaluate synthetic data
    synth_results = evaluate_synthetic(
        predictor, args.synthetic_path, args.device,
        checkpoint_dir=output_dir, batch_size=args.batch_size)
    print_summary(synth_results, "Synthetic Data")

    # Evaluate real data (optional baseline)
    real_results = []
    if args.real_path:
        real_results = evaluate_real(
            predictor, args.real_path, args.device,
            checkpoint_dir=output_dir, batch_size=args.batch_size)
        print_summary(real_results, "Real Data (reference)")

    # Save results
    def _dice_summary(res):
        d = np.array([r['dice'] for r in res])
        q25, q75 = np.percentile(d, [25, 75])
        return {
            'n_samples': len(res),
            'median_dice': float(np.median(d)),
            'q25_dice': float(q25),
            'q75_dice': float(q75),
            'results': res,
        }

    output = {
        'synthetic': _dice_summary(synth_results) if synth_results else {
            'n_samples': 0, 'median_dice': 0, 'q25_dice': 0, 'q75_dice': 0, 'results': []},
    }
    if real_results:
        output['real'] = _dice_summary(real_results)

    results_path = output_dir / 'dice_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
