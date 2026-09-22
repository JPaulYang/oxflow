"""
Flow-matching scheduler for diffusion training (JiT-style).

Continuous-time flow-matching formulation:
  Forward:  z = t*x + (1-t)*eps   (t=1 clean, t=0 noise)
  Target:   velocity v = (x - z) / (1-t)
  Sampling: ODE from t=0 to t=1 using Euler or Heun solver

Reference: "Back to Basics: Let Denoising Generative Models Denoise" (Li & He, 2025)
"""

import torch


class FlowMatchingScheduler:
    """Lightweight flow-matching scheduler — no learned parameters."""

    def __init__(self, P_mean=-0.8, P_std=0.8, t_eps=0.05, t_scale=1000.0):
        self.P_mean = P_mean      # sigmoid distribution mean
        self.P_std = P_std        # sigmoid distribution std
        self.t_eps = t_eps        # clamp (1-t) to avoid div-by-zero
        self.t_scale = t_scale    # scale t for UNet sinusoidal embedding

    # ---- training helpers ------------------------------------------------

    def sample_t(self, n, device):
        """Sample t in (0,1) from sigmoid(N(P_mean, P_std))."""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def add_noise(self, x, noise, t):
        """Flow-matching forward process: z = t*x + (1-t)*noise."""
        t = self._expand(t, x.ndim)
        return t * x + (1 - t) * noise

    def get_velocity(self, x, z, t):
        """Target velocity: v = (x - z) / (1-t)."""
        t = self._expand(t, x.ndim)
        return (x - z) / (1 - t).clamp_min(self.t_eps)

    def predict_velocity(self, x_pred, z, t):
        """Predicted velocity from model output x_pred."""
        return self.get_velocity(x_pred, z, t)

    # ---- sampling helpers ------------------------------------------------

    def scale_timestep(self, t):
        """Scale continuous t to range expected by UNet sinusoidal embedding."""
        return t * self.t_scale

    # ---- internal --------------------------------------------------------

    @staticmethod
    def _expand(t, ndim):
        """Expand t to broadcast with (B, C, H, W)."""
        while t.dim() < ndim:
            t = t.unsqueeze(-1)
        return t
