"""
ControlNet implementation for CT inpainting.

Architecture:
- ControlNet encoder mirrors the UNet encoder structure
- Zero convolutions connect ControlNet features to UNet skip connections
- Conditioning input: mask (1ch) + background_ct (1ch) = 2 channels
"""

import torch
import torch.nn as nn
from diffusers import UNet2DModel
from typing import Tuple, Optional


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ControlNetConditioningEmbedding(nn.Module):
    """
    Embedding for conditioning images (mask + background).
    Projects conditioning to match UNet's initial feature channels.
    """
    def __init__(
        self,
        conditioning_channels: int = 2,  # mask + background
        block_out_channels: Tuple[int, ...] = (64, 128, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            conditioning_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1
        )

        # Additional conv blocks to match UNet structure
        self.blocks = nn.ModuleList([])

        # Initial projection
        self.blocks.append(nn.Sequential(
            nn.Conv2d(block_out_channels[0], block_out_channels[0], kernel_size=3, padding=1),
            nn.SiLU(),
        ))

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conditioning: (B, 2, H, W) - mask + background concatenated

        Returns:
            (B, C, H, W) - embedded conditioning features
        """
        x = self.conv_in(conditioning)
        for block in self.blocks:
            x = block(x)
        return x


class ControlNetEncoder(nn.Module):
    """
    Encoder that mirrors UNet2DModel's downsampling structure.
    Extracts multi-scale features from conditioning input.
    """
    def __init__(
        self,
        conditioning_channels: int = 2,
        block_out_channels: Tuple[int, ...] = (64, 128, 256),
        layers_per_block: int = 2,
    ):
        super().__init__()

        self.conditioning_embedding = ControlNetConditioningEmbedding(
            conditioning_channels=conditioning_channels,
            block_out_channels=block_out_channels,
        )

        # Build encoder blocks matching UNet structure
        self.down_blocks = nn.ModuleList([])
        self.zero_convs = nn.ModuleList([])

        in_channels = block_out_channels[0]

        for i, out_channels in enumerate(block_out_channels):
            is_final_block = i == len(block_out_channels) - 1

            # Conv blocks for this level
            block_layers = []
            for j in range(layers_per_block):
                ch_in = in_channels if j == 0 else out_channels
                block_layers.append(nn.Sequential(
                    nn.Conv2d(ch_in, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                ))
                # Zero conv after each residual block
                self.zero_convs.append(zero_module(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                ))

            # Downsample (except for last block)
            if not is_final_block:
                block_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))
                self.zero_convs.append(zero_module(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                ))

            self.down_blocks.append(nn.ModuleList(block_layers))
            in_channels = out_channels

        # Middle block
        self.mid_block = nn.Sequential(
            nn.Conv2d(block_out_channels[-1], block_out_channels[-1], kernel_size=3, padding=1),
            nn.GroupNorm(8, block_out_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(block_out_channels[-1], block_out_channels[-1], kernel_size=3, padding=1),
            nn.GroupNorm(8, block_out_channels[-1]),
            nn.SiLU(),
        )
        self.mid_zero_conv = zero_module(
            nn.Conv2d(block_out_channels[-1], block_out_channels[-1], kernel_size=1)
        )

    def forward(self, conditioning: torch.Tensor) -> Tuple[list, torch.Tensor]:
        """
        Args:
            conditioning: (B, 2, H, W) - mask + background

        Returns:
            down_block_features: list of tensors for skip connections
            mid_block_feature: tensor for middle block
        """
        x = self.conditioning_embedding(conditioning)

        down_block_features = []
        zero_conv_idx = 0

        for block_layers in self.down_blocks:
            for layer in block_layers:
                x = layer(x)
                # Apply zero conv and store feature
                down_block_features.append(self.zero_convs[zero_conv_idx](x))
                zero_conv_idx += 1

        # Middle block
        mid = self.mid_block(x)
        mid_feature = self.mid_zero_conv(mid)

        return down_block_features, mid_feature


class ControlledUNet2D(nn.Module):
    """
    UNet2DModel wrapper that accepts ControlNet features.

    For inpainting:
    - UNet processes noisy CT image
    - ControlNet processes mask + background
    - Features are added at corresponding resolution levels
    """
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 1,  # noisy CT only
        out_channels: int = 1,
        block_out_channels: Tuple[int, ...] = (64, 128, 256),
        layers_per_block: int = 2,
        conditioning_channels: int = 2,  # mask + background
    ):
        super().__init__()

        # Main UNet (processes noisy image)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=("DownBlock2D",) * len(block_out_channels),
            up_block_types=("UpBlock2D",) * len(block_out_channels),
        )

        # ControlNet encoder (processes conditioning)
        self.controlnet = ControlNetEncoder(
            conditioning_channels=conditioning_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block

    def forward(
        self,
        noisy_sample: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: torch.Tensor,
        controlnet_scale: float = 1.0,
    ):
        """
        Args:
            noisy_sample: (B, 1, H, W) - noisy CT image
            timestep: (B,) or scalar - diffusion timestep
            conditioning: (B, 2, H, W) - mask + background concatenated
            controlnet_scale: scaling factor for controlnet features

        Returns:
            UNet2DOutput with sample attribute containing noise prediction
        """
        # Get controlnet features
        down_features, mid_feature = self.controlnet(conditioning)

        # Scale features
        down_features = [f * controlnet_scale for f in down_features]
        mid_feature = mid_feature * controlnet_scale

        # Forward through UNet with added features
        # Note: We need to manually add features during forward pass
        # This requires accessing UNet internals
        return self._forward_with_control(noisy_sample, timestep, down_features, mid_feature)

    def _forward_with_control(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        down_features: list,
        mid_feature: torch.Tensor,
    ):
        """Forward pass with controlnet feature injection."""
        unet = self.unet

        # Time embedding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(sample.shape[0])

        t_emb = unet.time_proj(timestep)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = unet.time_embedding(t_emb)

        # Pre-process
        sample = unet.conv_in(sample)

        # Down blocks with controlnet feature addition
        down_block_res_samples = (sample,)
        feature_idx = 0

        for down_block in unet.down_blocks:
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=emb,
                )
            else:
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=emb,
                )

            # Add controlnet features to residual samples
            res_samples_with_control = []
            for res in res_samples:
                if feature_idx < len(down_features):
                    ctrl_feat = down_features[feature_idx]
                    # Ensure shapes match
                    if ctrl_feat.shape == res.shape:
                        res = res + ctrl_feat
                    feature_idx += 1
                res_samples_with_control.append(res)

            down_block_res_samples += tuple(res_samples_with_control)

        # Middle block with controlnet feature addition
        if unet.mid_block is not None:
            sample = unet.mid_block(sample, emb)
            # Add mid feature
            if mid_feature.shape == sample.shape:
                sample = sample + mid_feature

        # Up blocks
        for up_block in unet.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                sample = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )
            else:
                sample = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )

        # Post-process
        sample = unet.conv_norm_out(sample)
        sample = unet.conv_act(sample)
        sample = unet.conv_out(sample)

        # Return in same format as UNet2DModel
        from diffusers.models.unets.unet_2d import UNet2DOutput
        return UNet2DOutput(sample=sample)

    def save_pretrained(self, save_directory: str):
        """Save both UNet and ControlNet weights."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        torch.save({
            'unet': self.unet.state_dict(),
            'controlnet': self.controlnet.state_dict(),
        }, os.path.join(save_directory, 'controlnet_model.pth'))

    def load_pretrained(self, load_directory: str, device: str = 'cpu'):
        """Load both UNet and ControlNet weights."""
        import os
        checkpoint = torch.load(
            os.path.join(load_directory, 'controlnet_model.pth'),
            map_location=device
        )
        self.unet.load_state_dict(checkpoint['unet'])
        self.controlnet.load_state_dict(checkpoint['controlnet'])


def create_controlnet_inpaint(
    image_size: int = 256,
    block_out_channels: Tuple[int, ...] = (64, 128, 256),
) -> ControlledUNet2D:
    """
    Create a ControlNet-based inpainting model.

    Args:
        image_size: Size of input images (assumed square)
        block_out_channels: Channel dimensions for each UNet block

    Returns:
        ControlledUNet2D model
    """
    return ControlledUNet2D(
        image_size=image_size,
        in_channels=1,  # noisy CT
        out_channels=1,  # noise prediction
        block_out_channels=block_out_channels,
        layers_per_block=2,
        conditioning_channels=2,  # mask + background
    )
