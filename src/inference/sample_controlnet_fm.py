"""
Sample from the mask-conditioned ControlNet with a flow-matching ODE solver.

The model generates the entire image (tumor + background) from the
conditioning (target mask + background CT with the tumor region filled with
its mean); the background is not re-imposed during sampling.
"""

import torch
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
import argparse

from src.models.controlnet import create_controlnet_inpaint
from src.models.flow_matching import FlowMatchingScheduler
from src.data.dataset_slices import LongitudinalCTDataset
from src.ot.interpolation import compute_ot_interpolated_mask


def sample_fm(
    model, scheduler, background, mask,
    steps=50, method='heun', controlnet_scale=1.0,
):
    """
    Flow-matching ODE sampling (x-prediction, Euler or Heun).

    The model freely generates the whole image; no background replacement
    is applied during the ODE integration.

    Args:
        model: ControlledUNet2D (predicts clean image x_pred)
        scheduler: FlowMatchingScheduler
        background: (B, 1, H, W) clean background CT (used only for conditioning)
        mask: (B, 1, H, W) binary tumor mask
        steps: number of ODE solver steps
        method: 'euler' or 'heun'
        controlnet_scale: ControlNet feature scaling
    """
    device = background.device
    B = background.shape[0]
    t_eps = scheduler.t_eps

    # Create conditioning (mask + masked_bg)
    bg_fill = background.mean(dim=(2, 3), keepdim=True).expand_as(background)
    masked_bg = background * (1 - mask) + bg_fill * mask
    cond = torch.cat([mask, masked_bg], dim=1)

    # Start from pure noise at t=0
    z = torch.randn_like(background)

    # Timesteps: 0 → 1 (noise → clean)
    timesteps = torch.linspace(0.0, 1.0, steps + 1, device=device)

    def model_velocity(z_in, t_val):
        """Compute velocity from model prediction at scalar time t_val."""
        t_batch = torch.full((B,), t_val, device=device)
        t_scaled = scheduler.scale_timestep(t_batch)
        x_pred = model(z_in, t_scaled, cond, controlnet_scale=controlnet_scale).sample
        v = (x_pred - z_in) / max(1.0 - t_val, t_eps)
        return v

    with torch.no_grad():
        for i in tqdm(range(steps), desc=f"Sampling FM ({method})"):
            t_cur = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t_cur

            if method == 'heun' and i < steps - 1:
                v1 = model_velocity(z, t_cur)
                z_euler = z + dt * v1
                v2 = model_velocity(z_euler, t_next)
                z = z + dt * 0.5 * (v1 + v2)
            else:
                v = model_velocity(z, t_cur)
                z = z + dt * v

    return z


def sample(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data_path = 'data_processed/longitudinal_data.pt'
    dataset = LongitudinalCTDataset(data_path, mode='eval', filter_no_tumor=True)

    patients = dataset.get_available_patients()[:3]
    print(f"Patients: {patients}")

    # Load model
    model = create_controlnet_inpaint(image_size=256).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)

    if args.use_ema and 'ema_state_dict' in ckpt:
        print("Loading EMA weights")
        model.load_state_dict(ckpt['ema_state_dict'])
    else:
        model.unet.load_state_dict(ckpt['unet_state_dict'])
        model.controlnet.load_state_dict(ckpt['controlnet_state_dict'])
    model.eval()

    scheduler = FlowMatchingScheduler(t_eps=args.t_eps)

    interp_ts = [0.25, 0.5, 0.75]
    ncols = 2 + len(interp_ts)  # Real t0 + interpolations + Real t1

    results = []
    for pid in patients:
        slices = [s for s in dataset.get_patient_slices(pid) if len(s['timepoints']) >= 2]
        s = slices[len(slices) // 2]

        tumor_sizes = [m.sum().item() for m in s['mask_series']]
        bg_t = min(range(len(tumor_sizes)), key=lambda i: tumor_sizes[i])
        other_t = len(tumor_sizes) - 1 if bg_t == 0 else 0

        ct_bg = s['ct_series'][bg_t].to(device)
        ct_other = s['ct_series'][other_t].to(device)
        mask_bg = s['mask_series'][bg_t]
        mask_other = s['mask_series'][other_t]

        print(f"  Patient {pid}: bg=t{bg_t} (tumor {tumor_sizes[bg_t]:.0f}px), "
              f"other=t{other_t} (tumor {tumor_sizes[other_t]:.0f}px)")

        interp_masks = []
        for t_val in interp_ts:
            m = compute_ot_interpolated_mask(mask_bg, mask_other, t=t_val).to(device)
            interp_masks.append(m)

        masks_batch = torch.stack(interp_masks)
        bgs_batch = ct_bg.unsqueeze(0).expand_as(masks_batch)

        gen_batch = sample_fm(
            model, scheduler, bgs_batch, masks_batch,
            steps=args.steps, method=args.method, controlnet_scale=args.controlnet_scale,
        )

        results.append({
            'ct_bg': ct_bg, 'ct_other': ct_other,
            'mask_bg': mask_bg.to(device), 'mask_other': mask_other.to(device),
            'interp_masks': interp_masks,
            'gen_interps': [gen_batch[i] for i in range(len(interp_ts))],
        })

    # Visualization
    def normalize(x):
        return ((x + 1) / 2).clamp(0, 1)

    def overlay(ct, mask, color=(0, 1, 0)):
        ct = normalize(ct)
        g = ct[0]
        m = (mask[0] > 0.5).float()
        kernel = torch.ones(1, 1, 3, 3, device=m.device)
        dilated = torch.nn.functional.conv2d(
            m.unsqueeze(0).unsqueeze(0), kernel, padding=1
        ).squeeze()
        contour = ((dilated > 0).float() - m).clamp(0, 1)
        r = g * (1 - contour) + color[0] * contour
        gr = g * (1 - contour) + color[1] * contour
        b = g * (1 - contour) + color[2] * contour
        return torch.stack([r, gr, b], dim=0)

    # Grid with mask overlay
    grid = []
    for r in results:
        grid.append(overlay(r['ct_bg'], r['mask_bg']))
        for i in range(len(interp_ts)):
            grid.append(overlay(r['gen_interps'][i], r['interp_masks'][i]))
        grid.append(overlay(r['ct_other'], r['mask_other']))
    grid = torch.stack(grid)

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_image(grid, f"{output_dir}/fm_samples_{len(patients)}x{ncols}.png", nrow=ncols, padding=2)
    print(f"Saved to {output_dir}/fm_samples_{len(patients)}x{ncols}.png")

    # Grid without mask overlay
    grid_clean = []
    for r in results:
        grid_clean.append(normalize(r['ct_bg']).expand(3, -1, -1))
        for gen in r['gen_interps']:
            grid_clean.append(normalize(gen).expand(3, -1, -1))
        grid_clean.append(normalize(r['ct_other']).expand(3, -1, -1))
    grid_clean = torch.stack(grid_clean)
    save_image(grid_clean, f"{output_dir}/fm_samples_{len(patients)}x{ncols}_clean.png", nrow=ncols, padding=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='./outputs/controlnet_fm_samples')
    p.add_argument('--steps', type=int, default=50, help='ODE solver steps (default 50)')
    p.add_argument('--method', type=str, default='heun', choices=['euler', 'heun'],
                   help='ODE solver method')
    p.add_argument('--controlnet_scale', type=float, default=1.5)
    p.add_argument('--t_eps', type=float, default=0.05)
    p.add_argument('--use_ema', action='store_true', default=True,
                   help='Use EMA weights if available')
    p.add_argument('--no_ema', dest='use_ema', action='store_false')
    args = p.parse_args()
    sample(args)
