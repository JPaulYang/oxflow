"""
Convert longitudinal_data.pt to nnUNet v2 raw data format for 2D tumor segmentation.

Extracts tumor-bearing slices (each timepoint treated as an independent 2D case)
and saves as NIfTI files in the nnUNet raw data directory structure.

Prerequisite:
    pip install nnunetv2 nibabel

Usage:
    # Set nnUNet environment variables first
    export nnUNet_raw="data_processed/nnUNet_raw"
    export nnUNet_preprocessed="data_processed/nnUNet_preprocessed"
    export nnUNet_results="data_processed/nnUNet_results"

    python script/train/prepare_nnunet_data.py \
        --data_path data_processed/longitudinal_data.pt \
        --dataset_id 1 \
        --dataset_name LungTumor2D

After running this script:
    # Verify and preprocess (2D config since our data is 2D slices)
    nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 2d

    # Train (fold 0)
    nnUNetv2_train 1 2d 0
"""

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm


TUMOR_MIN_PIXELS = 3


def tensor_to_nifti_2d(tensor_2d, save_path):
    """
    Save a 2D tensor as a 3D NIfTI with shape (H, W, 1).
    nnUNet expects 3D volumes; for 2D we use a single-slice volume.
    """
    arr = tensor_2d.numpy().astype(np.float32)
    # (H, W) -> (H, W, 1) for NIfTI
    arr_3d = arr[:, :, np.newaxis]
    # Use identity affine with 1mm spacing
    affine = np.eye(4)
    img = nib.Nifti1Image(arr_3d, affine)
    nib.save(img, str(save_path))


def main():
    parser = argparse.ArgumentParser(
        description='Convert longitudinal_data.pt to nnUNet v2 raw format')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to longitudinal_data.pt')
    parser.add_argument('--dataset_id', type=int, default=1,
                        help='nnUNet dataset ID (e.g., 1 -> Dataset001_...)')
    parser.add_argument('--dataset_name', type=str, default='LungTumor2D',
                        help='Dataset name suffix')
    parser.add_argument('--min_tumor_pixels', type=int, default=TUMOR_MIN_PIXELS,
                        help='Minimum tumor pixels to include a slice')
    args = parser.parse_args()

    # Resolve output directory from nnUNet_raw env var
    nnunet_raw = os.environ.get('nnUNet_raw')
    if nnunet_raw is None:
        print("ERROR: nnUNet_raw environment variable not set.")
        print("Run: export nnUNet_raw=\"data_processed/nnUNet_raw\"")
        return

    dataset_dir_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_dir = Path(nnunet_raw) / dataset_dir_name
    images_dir = dataset_dir / 'imagesTr'
    labels_dir = dataset_dir / 'labelsTr'

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {dataset_dir}")

    # Load data
    print(f"Loading data from {args.data_path}")
    all_data = torch.load(args.data_path)
    print(f"Total entries: {len(all_data)}")

    case_id = 0
    skipped = 0
    case_mapping = []  # Track case_id -> (patient_id, slice_idx, timepoint)

    for item in tqdm(all_data, desc="Converting slices"):
        patient_id = item['patient_id']
        slice_idx = item['slice_idx']
        ct_slices = item['ct_slices']     # (T, 1, H, W)
        masks = item['masks']             # (T, 1, H, W)
        T = ct_slices.shape[0]

        for t in range(T):
            mask_t = (masks[t, 0] > 0).float()  # (H, W)

            # Only include slices with sufficient tumor
            if mask_t.sum() < args.min_tumor_pixels:
                skipped += 1
                continue

            ct_t = ct_slices[t, 0]  # (H, W)

            # nnUNet naming: case_XXXXX_0000.nii.gz (0000 = channel 0)
            case_name = f"case_{case_id:05d}"
            image_path = images_dir / f"{case_name}_0000.nii.gz"
            label_path = labels_dir / f"{case_name}.nii.gz"

            tensor_to_nifti_2d(ct_t, image_path)
            tensor_to_nifti_2d(mask_t, label_path)

            case_mapping.append({
                "case_id": case_name,
                "patient_id": patient_id,
                "slice_idx": int(slice_idx),
                "timepoint": t,
            })
            case_id += 1

    print(f"\nConverted {case_id} tumor-bearing slices ({skipped} skipped)")

    # Save case mapping for traceability
    mapping_path = dataset_dir / 'case_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(case_mapping, f, indent=2)
    print(f"Case mapping saved to {mapping_path}")

    # Generate dataset.json
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "tumor": 1
        },
        "numTraining": case_id,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }

    dataset_json_path = dataset_dir / 'dataset.json'
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"dataset.json saved to {dataset_json_path}")
    print(f"\nNext steps:")
    print(f"  1. export nnUNet_raw=\"{Path(nnunet_raw).resolve()}\"")
    print(f"  2. export nnUNet_preprocessed=\"data_processed/nnUNet_preprocessed\"")
    print(f"  3. export nnUNet_results=\"data_processed/nnUNet_results\"")
    print(f"  4. nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity -c 2d")
    print(f"  5. nnUNetv2_train {args.dataset_id} 2d 0")


if __name__ == '__main__':
    main()
