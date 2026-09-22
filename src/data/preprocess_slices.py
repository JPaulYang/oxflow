import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Read NIfTI file and transform to 2D slices
def load_nii(path):
    img = nib.load(str(path))
    arr = img.get_fdata().astype(np.float32) # shape: (W, D, H); fdata - get data as float32
    return arr


def normalize_ct(x, min_hu=-1000, max_hu=400):
    x = np.clip(x, min_hu, max_hu)
    x = (x - min_hu) / (max_hu - min_hu)
    x = x * 2 - 1
    return x


def preprocess_scan(ct_path, mask_path, out_h=256, out_w=256):
    """
    Process a single scan (one timepoint) and return ALL slices.
    Returns:
        ct_slices: (H, 1, out_h, out_w) - all slices along the z-axis
        mask_slices: (H, 1, out_h, out_w)
        valid_mask: (H,) - boolean mask indicating which slices have tumor
    """
    ct = load_nii(ct_path) # (W, D, H), np.float32
    mask = load_nii(mask_path) # (W, D, H)
    ct = normalize_ct(ct)

    W, D, H = ct.shape # (R, A, S)
    ct_slices = []
    mask_slices = []
    valid_mask = []

    for z in range(H):
        ct_slice = ct[:, :, z]
        mask_slice = mask[:, :, z]

        # Rotate 90 degrees counterclockwise (k=1 means 90 degrees CCW)
        # Need .copy() because rot90 creates a view with negative strides
        ct_slice = np.rot90(ct_slice, k=1).copy()
        mask_slice = np.rot90(mask_slice, k=1).copy()

        # Track if this slice has tumor
        has_tumor = bool(mask_slice.max() > 0)
        valid_mask.append(has_tumor)

        ct_t = torch.from_numpy(ct_slice).unsqueeze(0).unsqueeze(0) # (1, 1, W, D)
        mask_t = torch.from_numpy(mask_slice).unsqueeze(0).unsqueeze(0) # (1, 1, W, D)

        ct_t = F.interpolate(ct_t, size=(out_h, out_w), mode='bilinear', align_corners=False)
        mask_t = F.interpolate(mask_t, size=(out_h, out_w), mode='nearest')

        ct_slices.append(ct_t.squeeze(0)) # (1, out_h, out_w)
        mask_slices.append(mask_t.squeeze(0)) # (1, out_h, out_w)

    ct_slices = torch.stack(ct_slices, dim=0) # (H, 1, out_h, out_w)
    mask_slices = torch.stack(mask_slices, dim=0) # (H, 1, out_h, out_w)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool) # (H,)

    return ct_slices, mask_slices, valid_mask


def extract_timepoint_from_filename(filename):
    """
    Extract timepoint from filename.
    Example: 000123_1_scanA_img.nii.gz -> timepoint = 1
    Adjust this function based on your actual naming convention.
    """
    # Assuming format: {patient_id}_{timepoint}_{scan_id}_img.nii.gz
    match = re.search(r'_(\d+)_[A-Za-z0-9]+_img\.nii\.gz', filename)
    if match:
        return int(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Slice longitudinal NIfTI CT volumes + tumor masks into a longitudinal_data.pt tensor file')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Directory with one sub-directory per patient: <pid>/img/*_img.nii.gz, <pid>/msk/*_gt.nii.gz')
    parser.add_argument('--output', type=str, default='data_processed/longitudinal_data.pt')
    args = parser.parse_args()

    data_root = Path(args.data_root)

    # get all patient IDs
    patients_ids = sorted([d.name for d in data_root.iterdir() if d.is_dir()])

    # Data structure to store longitudinal slices
    # Structure: list of dicts, each dict represents one (patient, slice_position) pair
    # {
    #   'patient_id': int,
    #   'slice_idx': int,  # z-position in the original volume
    #   'timepoints': [int, ...],  # list of timepoint indices
    #   'ct_slices': tensor (T, 1, H, W),  # T = number of timepoints
    #   'masks': tensor (T, 1, H, W),
    #   'has_tumor': bool,
    #   'has_tumor_per_timepoint': [bool, ...],
    # }
    longitudinal_data = []

    for patient_id in tqdm(patients_ids, desc='Processing patients...'):
        img_dir = data_root / patient_id / 'img'
        msk_dir = data_root / patient_id / 'msk'

        # get all CT files and sort by timepoint
        ct_files = sorted(img_dir.glob('*_img.nii.gz'))

        if len(ct_files) == 0:
            continue

        # Process all scans for this patient
        patient_scans = []
        for ct_file in ct_files:
            mask_file = msk_dir / ct_file.name.replace('_img.nii.gz', '_gt.nii.gz')

            if not mask_file.exists():
                print(f"Warning: mask file not found for {ct_file}")
                continue

            timepoint = extract_timepoint_from_filename(ct_file.name)
            if timepoint is None:
                print(f"Warning: could not extract timepoint from {ct_file.name}")
                continue

            ct_slices, mask_slices, valid_mask = preprocess_scan(ct_file, mask_file)

            patient_scans.append({
                'timepoint': timepoint,
                'ct': ct_slices,
                'mask': mask_slices,
                'valid': valid_mask
            })

        if len(patient_scans) == 0:
            continue

        # Sort by timepoint
        patient_scans = sorted(patient_scans, key=lambda x: x['timepoint'])

        # Assume all scans have the same number of slices (same z-dimension)
        assert all(scan['ct'].shape[0] == patient_scans[0]['ct'].shape[0] for scan in patient_scans), \
            "Mismatch in number of slices across timepoints"
        # If not, you may need to implement registration/alignment
        num_slices = patient_scans[0]['ct'].shape[0]

        # For each slice position, collect data across all timepoints
        for z in range(num_slices):
            # Check if at least one timepoint has tumor at this position
            has_any_tumor = any(scan['valid'][z] for scan in patient_scans)

            # Collect data for all slices, not just those with tumors
            timepoints = [scan['timepoint'] for scan in patient_scans]
            ct_at_z = torch.stack([scan['ct'][z] for scan in patient_scans], dim=0)  # (T, 1, H, W)
            mask_at_z = torch.stack([scan['mask'][z] for scan in patient_scans], dim=0)  # (T, 1, H, W)

            # Track which timepoints have tumor at this position
            has_tumor_per_timepoint = [scan['valid'][z] for scan in patient_scans]

            pid_int = int(patient_id)
            longitudinal_data.append({
                'patient_id': pid_int,
                'slice_idx': z,
                'timepoints': timepoints,
                'ct_slices': ct_at_z,
                'masks': mask_at_z,
                'has_tumor': has_any_tumor,  # True if ANY timepoint has tumor
                'has_tumor_per_timepoint': has_tumor_per_timepoint,  # List of bools for each timepoint
            })

    # Save the processed data
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(longitudinal_data, out_path)
    print(f"Saved to {out_path}")

    print(f"Processed {len(patients_ids)} patients")
    print(f"Total longitudinal slice positions: {len(longitudinal_data)}")

    # Print some statistics
    timepoint_counts = {}
    slices_with_tumor = 0
    slices_without_tumor = 0

    for item in longitudinal_data:
        n_timepoints = len(item['timepoints'])
        timepoint_counts[n_timepoints] = timepoint_counts.get(n_timepoints, 0) + 1

        if item['has_tumor']:
            slices_with_tumor += 1
        else:
            slices_without_tumor += 1

    print(f"\nSlices with tumor (at least one timepoint): {slices_with_tumor}")
    print(f"Slices without tumor (all timepoints): {slices_without_tumor}")

    print("\nDistribution of timepoints per slice position:")
    for n, count in sorted(timepoint_counts.items()):
        print(f"  {n} timepoints: {count} positions")


if __name__ == '__main__':
    main()

