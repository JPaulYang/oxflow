"""
Generate a 3-row × 5-column qualitative comparison figure.

Layout (columns: t0, α=0.25, α=0.5, α=0.75, t1):
  Row 1: Mask only — real mask at t0, OT-interpolated masks, real mask at t1
  Row 2: OT mask + FM (Ours) — real CT at t0, generated CTs, real CT at t1
  Row 3: Random mask + FM — real CT at t0, generated CTs, real CT at t1

Usage:
    python script/visualize/visualize_method_comparison.py \
        --patient_id <PATIENT_ID> \
        --fm_ckpt path/to/controlnet_fm.pth
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from src.data.dataset_slices import LongitudinalCTDataset
from src.models.controlnet import create_controlnet_inpaint
from src.models.flow_matching import FlowMatchingScheduler
from src.inference.sample_controlnet_fm import sample_fm
from src.ot.interpolation import compute_ot_interpolated_mask, compute_random_interpolated_mask


# ─── Helpers ─────────────────────────────────────────────────────────────────

def normalize(x):
    """[-1, 1] -> [0, 1]."""
    return ((x + 1) / 2).clamp(0, 1)


def mask_to_rgb(mask):
    """Convert binary mask (1, H, W) to white-on-black RGB (3, H, W)."""
    m = (mask[0] > 0.5).float()
    return torch.stack([m, m, m], dim=0)


def ct_to_rgb(ct):
    """Grayscale CT (1, H, W) to RGB (3, H, W). Fixed [-1, 1] -> [0, 1]."""
    g = normalize(ct)[0]
    return torch.stack([g, g, g], dim=0)


def overlay_contour(ct_rgb, mask, color=(0, 1, 0), linewidth=1):
    """Draw mask contour on RGB image. Returns (3, H, W)."""
    m = (mask[0] > 0.5).float()
    if m.max() < 0.5:
        return ct_rgb.clone()
    device = m.device
    kernel = torch.ones(1, 1, 3, 3, device=device)
    dilated = torch.nn.functional.conv2d(
        m.unsqueeze(0).unsqueeze(0), kernel, padding=1
    ).squeeze()
    contour = ((dilated > 0).float() - m).clamp(0, 1)
    # Thicken if requested
    for _ in range(linewidth - 1):
        dilated2 = torch.nn.functional.conv2d(
            contour.unsqueeze(0).unsqueeze(0), kernel, padding=1
        ).squeeze()
        contour = (dilated2 > 0).float().clamp(0, 1)
    out = ct_rgb.clone()
    for c in range(3):
        out[c] = out[c] * (1 - contour) + color[c] * contour
    return out


def get_patient_data(dataset, pid, device):
    """Get middle tumor slice for a patient. Returns data dict or None."""
    slices = [s for s in dataset.get_patient_slices(pid)
              if len(s['timepoints']) >= 2]
    if not slices:
        return None
    s = slices[len(slices) // 2]

    n_tp = len(s['mask_series'])
    tumor_sizes = [m.sum().item() for m in s['mask_series']]
    # OT direction: smaller → larger tumor
    bg_idx = min(range(n_tp), key=lambda i: tumor_sizes[i])
    tgt_idx = (n_tp - 1) if bg_idx == 0 else 0

    return {
        'ct_first': s['ct_series'][0].to(device),
        'ct_last': s['ct_series'][-1].to(device),
        'mask_first': s['mask_series'][0].to(device),
        'mask_last': s['mask_series'][-1].to(device),
        'ct_bg': s['ct_series'][bg_idx].to(device),
        'mask_bg': s['mask_series'][bg_idx].to(device),
        'mask_tgt': s['mask_series'][tgt_idx].to(device),
        'reversed': bg_idx != 0,
        'tumor_sizes': tumor_sizes,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate 3×5 method comparison figure')
    parser.add_argument('--patient_id', type=int, required=True,
                        help='Patient ID to visualize')
    parser.add_argument('--data_path', type=str,
                        default='data_processed/longitudinal_data.pt')
    parser.add_argument('--fm_ckpt', type=str,
                        required=True,
                        help='Flow matching checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/figures')
    parser.add_argument('--fm_steps', type=int, default=50)
    parser.add_argument('--controlnet_scale', type=float, default=1.5)
    parser.add_argument('--t_eps', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smooth_sigma', type=float, default=0,
                        help='Gaussian sigma for mask edge smoothing (0=off)')
    parser.add_argument('--convex_hull', action='store_true', default=True,
                        help='Use convex hull for OT mask visualization (default: on)')
    parser.add_argument('--no_convex_hull', dest='convex_hull', action='store_false',
                        help='Disable convex hull, use closing + optional smooth_sigma')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = LongitudinalCTDataset(args.data_path, mode='eval',
                                     filter_no_tumor=True)
    data = get_patient_data(dataset, args.patient_id, device)
    if data is None:
        print(f"Patient {args.patient_id} not found or has <2 timepoints")
        return

    ct_bg = data['ct_bg']
    mask_bg, mask_tgt = data['mask_bg'], data['mask_tgt']
    is_reversed = data['reversed']
    direction = "shrinking" if is_reversed else "growing"
    print(f"Patient {args.patient_id}: tumor {direction}, "
          f"sizes: {[f'{s:.0f}' for s in data['tumor_sizes']]}")

    interp_ts = [0.25, 0.5, 0.75]

    # ── Compute masks ────────────────────────────────────────────────────
    mask_mode = "convex_hull" if args.convex_hull else f"smooth_sigma={args.smooth_sigma}"
    print(f"Computing OT-interpolated masks ({mask_mode})...")
    ot_masks = [compute_ot_interpolated_mask(
                    mask_bg, mask_tgt, t=t,
                    smooth_sigma=args.smooth_sigma,
                    convex_hull=args.convex_hull,
                ).to(device) for t in interp_ts]

    print("Computing random-matched masks (no smoothing)...")
    random_masks = [compute_random_interpolated_mask(
                        mask_bg, mask_tgt, t=t, smooth_sigma=0, convex_hull=False
                    ).to(device) for t in interp_ts]

    # Reverse for chronological display if needed
    if is_reversed:
        ot_masks_display = ot_masks[::-1]
        random_masks_display = random_masks[::-1]
    else:
        ot_masks_display = ot_masks
        random_masks_display = random_masks

    # ── Load FM model ────────────────────────────────────────────────────
    print("Loading FM model...")
    fm_model = create_controlnet_inpaint(image_size=256).to(device)
    fm_ckpt = torch.load(args.fm_ckpt, map_location=device)
    if 'ema_state_dict' in fm_ckpt:
        fm_model.load_state_dict(fm_ckpt['ema_state_dict'])
    else:
        fm_model.unet.load_state_dict(fm_ckpt['unet_state_dict'])
        fm_model.controlnet.load_state_dict(fm_ckpt['controlnet_state_dict'])
    fm_model.eval()
    fm_scheduler = FlowMatchingScheduler(t_eps=args.t_eps)

    # ── Generate OT + FM  ─────────────────────────────────────
    print("Generating: OT mask + FM ...")
    masks_batch = torch.stack(ot_masks)
    bgs_batch = ct_bg.unsqueeze(0).expand_as(masks_batch)
    gen_ot = sample_fm(
        fm_model, fm_scheduler, bgs_batch, masks_batch,
        steps=args.fm_steps, method='heun',
        controlnet_scale=1.0,
    )
    ot_gens = [gen_ot[i] for i in range(len(interp_ts))]
    if is_reversed:
        ot_gens = ot_gens[::-1]

    # ── Generate Random + FM  ─────────────────────────────────
    print("Generating: Random mask + FM ...")
    masks_batch = torch.stack(random_masks)
    bgs_batch = ct_bg.unsqueeze(0).expand_as(masks_batch)
    gen_random = sample_fm(
        fm_model, fm_scheduler, bgs_batch, masks_batch,
        steps=args.fm_steps, method='heun',
        controlnet_scale=1.0,
    )
    random_gens = [gen_random[i] for i in range(len(interp_ts))]
    if is_reversed:
        random_gens = random_gens[::-1]

    # ── Build figure ─────────────────────────────────────────────────────
    print("Building figure...")
    ct_first, ct_last = data['ct_first'], data['ct_last']
    mask_first, mask_last = data['mask_first'], data['mask_last']

    # Row 1: OT Masks (white on black)
    row_masks = [mask_to_rgb(mask_first)]
    for m in ot_masks_display:
        row_masks.append(mask_to_rgb(m))
    row_masks.append(mask_to_rgb(mask_last))

    # Random masks (white on black) — saved as individual images only
    row_random_masks = [mask_to_rgb(mask_first)]
    for m in random_masks_display:
        row_random_masks.append(mask_to_rgb(m))
    row_random_masks.append(mask_to_rgb(mask_last))

    # Row 2: OT + FM
    row_ot = [ct_to_rgb(ct_first)]
    for gen in ot_gens:
        row_ot.append(ct_to_rgb(gen))
    row_ot.append(ct_to_rgb(ct_last))

    # Row 3: Random + FM
    row_rand = [ct_to_rgb(ct_first)]
    for gen in random_gens:
        row_rand.append(ct_to_rgb(gen))
    row_rand.append(ct_to_rgb(ct_last))

    # ── Save individual cell images ─────────────────────────────────────
    method_names = ['mask', 'random_mask', 'ot_fm', 'random_fm']
    alpha_names = ['a000', 'a025', 'a050', 'a075', 'a100']
    rows_data = {
        'mask': row_masks,
        'random_mask': row_random_masks,
        'ot_fm': row_ot,
        'random_fm': row_rand,
    }

    single_dir = output_dir / 'single_img' / f'pid{args.patient_id}'
    single_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving individual images to {single_dir}/")

    from PIL import Image
    for method in method_names:
        for i, alpha in enumerate(alpha_names):
            cell = rows_data[method][i]  # (3, H, W) tensor
            arr = (cell.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(single_dir / f'{method}_{alpha}.png')

    print(f"Saved {len(method_names) * len(alpha_names)} individual images")

    # ── Build composite figure ────────────────────────────────────────
    rows = [row_masks, row_ot, row_rand]
    row_labels = [
        'OT masks',
        'OT mask + FM\n(Ours)',
        'Random mask\n+ FM',
    ]
    col_labels = [r'$\alpha=0$ (GT)', r'$\alpha=0.25$', r'$\alpha=0.50$',
                  r'$\alpha=0.75$', r'$\alpha=1$ (GT)']

    n_rows, n_cols = 3, 5

    # Stitch images into a single array with no gaps
    imgs = []
    for r in range(n_rows):
        row_imgs = []
        for c in range(n_cols):
            img = rows[r][c].detach().cpu().permute(1, 2, 0).numpy()
            row_imgs.append(img)
        imgs.append(np.concatenate(row_imgs, axis=1))  # concat along width
    canvas = np.concatenate(imgs, axis=0)  # concat along height

    H_img = imgs[0].shape[0]  # height of one image
    W_img = rows[0][0].shape[2]  # width of one image

    # Figure size: tight around the canvas
    aspect = canvas.shape[1] / canvas.shape[0]
    fig_h = 7.0
    fig_w = fig_h * aspect + 1.2  # extra space for row labels
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    ax.imshow(canvas)
    ax.axis('off')

    # Column labels at bottom
    for c in range(n_cols):
        x = W_img * (c + 0.5)
        y = canvas.shape[0] + 8
        ax.text(x, y, col_labels[c], ha='center', va='top', fontsize=11)

    # Row labels on the left
    for r in range(n_rows):
        y = H_img * (r + 0.5)
        ax.text(-10, y, row_labels[r], ha='right', va='center', fontsize=10)

    ax.set_xlim(-W_img * 0.35, canvas.shape[1])
    ax.set_ylim(canvas.shape[0] + 25, -5)

    for fmt in ['pdf', 'png']:
        out_path = output_dir / f'method_comparison_pid{args.patient_id}.{fmt}'
        fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
