"""
Train ControlNet for CT inpainting with Flow-Matching (JiT-style).

  - Continuous time t ∈ (0,1) sampled via sigmoid schedule
  - Forward process: z = t*x + (1-t)*ε   (linear interpolation)
  - Model predicts clean image x_pred (not noise)
  - Velocity prediction loss: MSE(v, v_pred)
  - EMA model tracking for stable sampling
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import copy
import argparse

from src.data import LongitudinalCTDataset
from src.models.controlnet import create_controlnet_inpaint
from src.models.flow_matching import FlowMatchingScheduler


def get_grad_norm(model):
    """Compute gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


@torch.no_grad()
def update_ema(ema_state, model_state, decay):
    """Update EMA state dict in-place."""
    for k in ema_state:
        ema_state[k].mul_(decay).add_(model_state[k], alpha=1 - decay)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = LongitudinalCTDataset(
        os.path.join(args.data_path, 'longitudinal_data.pt'),
        mode='train', filter_no_tumor=True
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Dataset: {len(dataset)} samples")

    model = create_controlnet_inpaint(image_size=args.image_size).to(device)
    scheduler = FlowMatchingScheduler(
        P_mean=args.P_mean,
        P_std=args.P_std,
        t_eps=args.t_eps,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Flow-matching: P_mean={args.P_mean}, P_std={args.P_std}, "
          f"t_eps={args.t_eps}, ema_decay={args.ema_decay}")

    ckpt_path = os.path.join(args.ckpt_path, 'controlnet_fm_latest.pth')
    start_epoch = 0
    global_step = 0
    wandb_id = args.wandb_resume

    # Initialize EMA state
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load checkpoint if exists
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.unet.load_state_dict(ckpt['unet_state_dict'])
        model.controlnet.load_state_dict(ckpt['controlnet_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'ema_state_dict' in ckpt:
            ema_state = ckpt['ema_state_dict']
        else:
            ema_state = {k: v.clone() for k, v in model.state_dict().items()}
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', start_epoch * len(loader))
        if wandb_id is None:
            wandb_id = ckpt.get('wandb_id')
        print(f"Resumed from epoch {start_epoch}")

    # Initialize wandb
    if args.wandb:
        import wandb
        run_name = args.wandb_run
        if run_name is None:
            run_name = f"fm_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"

        try:
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                id=wandb_id,
                resume="allow",
                config=vars(args),
                settings=wandb.Settings(init_timeout=120),
            )
            wandb_id = wandb.run.id
            print(f"Wandb run: {wandb.run.name} (id: {wandb_id})")
        except Exception as e:
            print(f"Wandb init failed: {e}, falling back to offline mode")
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                id=wandb_id,
                resume="allow",
                config=vars(args),
                mode="offline",
            )
            wandb_id = wandb.run.id
            print(f"Wandb offline run: {wandb.run.name} (id: {wandb_id})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_bg_loss = 0.0

        for ct, mask in pbar:
            ct, mask = ct.to(device), mask.to(device)

            # CRITICAL: masked_bg hides tumor info, forcing model to generate
            bg_fill = ct.mean(dim=(2, 3), keepdim=True).expand_as(ct)
            masked_bg = ct * (1 - mask) + bg_fill * mask

            # Conditioning: mask + masked_background
            cond = torch.cat([mask, masked_bg], dim=1)

            # ---- Flow-matching forward process ----
            noise = torch.randn_like(ct)
            t = scheduler.sample_t(ct.size(0), device=device)          # (B,)
            t_expanded = t.view(-1, 1, 1, 1)                           # (B,1,1,1)

            # Noisy sample: z = t*x + (1-t)*ε
            z = t_expanded * ct + (1 - t_expanded) * noise

            # Target velocity: v = (x - z) / (1-t)
            v_target = (ct - z) / (1 - t_expanded).clamp_min(args.t_eps)

            # Model predicts clean image x_pred
            t_scaled = scheduler.scale_timestep(t)                     # scale for UNet embedding
            x_pred = model(z, t_scaled, cond).sample

            # Predicted velocity: v_pred = (x_pred - z) / (1-t)
            v_pred = (x_pred - z) / (1 - t_expanded).clamp_min(args.t_eps)

            # ---- Velocity MSE loss with tumor weighting ----
            mse = (v_target - v_pred) ** 2
            mask_loss = (mse * mask).sum() / (mask.sum() + 1e-8)
            bg_loss = (mse * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)

            # Weighted loss: 2x on tumor region
            weight = 1.0 + mask
            loss = (weight * mse).mean()

            optimizer.zero_grad()
            loss.backward()
            grad_norm = get_grad_norm(model)
            optimizer.step()

            # Update EMA
            update_ema(ema_state, model.state_dict(), args.ema_decay)

            # Accumulate
            epoch_loss += loss.item()
            epoch_mask_loss += mask_loss.item()
            epoch_bg_loss += bg_loss.item()
            global_step += 1

            # Log to wandb
            if args.wandb and global_step % args.log_interval == 0:
                import wandb
                wandb.log({
                    'train/loss': loss.item(),
                    'train/mask_loss': mask_loss.item(),
                    'train/bg_loss': bg_loss.item(),
                    'train/grad_norm': grad_norm,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/t_mean': t.mean().item(),
                    'train/t_std': t.std().item(),
                }, step=global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm:.2f}")

        # Epoch summary
        n_batches = len(loader)
        avg_loss = epoch_loss / n_batches
        avg_mask = epoch_mask_loss / n_batches
        avg_bg = epoch_bg_loss / n_batches
        print(f"Epoch {epoch+1} - loss: {avg_loss:.4f}, mask: {avg_mask:.4f}, bg: {avg_bg:.4f}")

        if args.wandb:
            import wandb
            wandb.log({
                'epoch/loss': avg_loss,
                'epoch/mask_loss': avg_mask,
                'epoch/bg_loss': avg_bg,
                'epoch': epoch + 1,
            }, step=global_step)

        # Save checkpoint (includes EMA and wandb_id for resume)
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'unet_state_dict': model.unet.state_dict(),
            'controlnet_state_dict': model.controlnet.state_dict(),
            'ema_state_dict': ema_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'wandb_id': wandb_id if args.wandb else None,
        }, ckpt_path)

        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'unet_state_dict': model.unet.state_dict(),
                'controlnet_state_dict': model.controlnet.state_dict(),
                'ema_state_dict': ema_state,
            }, os.path.join(args.ckpt_path, f'controlnet_fm_epoch{epoch+1}.pth'))
            print(f"Saved epoch {epoch+1}")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # training
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--data_path', type=str, default='data_processed/')
    p.add_argument('--ckpt_path', type=str, default='ckpts/')
    p.add_argument('--save_interval', type=int, default=10)
    # flow-matching
    p.add_argument('--P_mean', type=float, default=-0.8, help='Sigmoid schedule mean')
    p.add_argument('--P_std', type=float, default=0.8, help='Sigmoid schedule std')
    p.add_argument('--t_eps', type=float, default=0.05, help='Clamp min for (1-t)')
    p.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate')
    # wandb
    p.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    p.add_argument('--wandb_project', type=str, default='ct-inpainting')
    p.add_argument('--wandb_run', type=str, default=None, help='Run name (auto-generated if not set)')
    p.add_argument('--wandb_resume', type=str, default=None, help='Wandb run id to resume')
    p.add_argument('--log_interval', type=int, default=10, help='Log every N steps')
    args = p.parse_args()
    os.makedirs(args.ckpt_path, exist_ok=True)
    train(args)
