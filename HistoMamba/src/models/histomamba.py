# src/models/histomamba.py

import torch
import torch.nn as nn
import math

# Try to import Mamba/DropPath but don't crash if they are missing initially
MAMBA_AVAILABLE = False
try:
    from mamba_ssm.modules.mamba_simple import Mamba as MambaSimple
    Mamba = MambaSimple
    MAMBA_AVAILABLE = True
    print("Imported Mamba from mamba_ssm.modules.mamba_simple.")
except ImportError:
    try:
        from mamba_ssm import Mamba as MambaSSM
        Mamba = MambaSSM
        MAMBA_AVAILABLE = True
        print("Imported Mamba from mamba_ssm (root).")
    except ImportError:
        print("\n********************************************************************")
        print("Warning: mamba_ssm (Mamba) not found. Install it (`pip install mamba-ssm causal-conv1d>=1.1.0`).")
        print("The HistoMamba model will fall back to a CNN-only architecture.")
        print("********************************************************************\n")
        Mamba = None

try:
    from timm.models.layers import DropPath
    print("Imported DropPath from timm.")
except ImportError:
    print("Warning: timm not found. Using nn.Identity instead of DropPath.")
    DropPath = lambda drop_prob=0.: nn.Identity()


class SpatialMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4.0,
                 dropout=0.1, drop_path=0.1,
                 sasf_kernel_size=3, sasf_dilations=[1, 3, 5]):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba/Mamba2 class not available. SpatialMambaBlock cannot be created.")
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.sasf_convs = nn.ModuleList()
        for dilation in sasf_dilations:
            padding = (sasf_kernel_size - 1) * dilation // 2
            self.sasf_convs.append(
                nn.Conv2d(dim, dim, kernel_size=sasf_kernel_size,
                          padding=padding, groups=dim, dilation=dilation, bias=True)
            )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim), nn.Dropout(dropout)
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut_flat = x.flatten(2).transpose(1, 2)
        x_flat = x.flatten(2).transpose(1, 2)
        normed_x_flat = self.norm1(x_flat)
        mamba_out_seq = self.mamba_fwd(normed_x_flat)
        state_proxy_2d = mamba_out_seq.transpose(1, 2).reshape(B, C, H, W)
        fused_state_2d = torch.zeros_like(state_proxy_2d)
        for conv in self.sasf_convs:
            fused_state_2d = fused_state_2d + conv(state_proxy_2d)
        fused_state_flat = fused_state_2d.flatten(2).transpose(1, 2)
        x_flat = shortcut_flat + self.drop_path1(fused_state_flat)
        x_flat = x_flat + self.drop_path2(self.mlp(self.norm2(x_flat)))
        out = x_flat.transpose(1, 2).reshape(B, C, H, W)
        return out


class HistoMamba(nn.Module):
    def __init__(self, num_classes=1, img_size=256, dims=[64, 128, 256, 512],
                 use_lpu=False, drop_path_rate=0.1, **block_kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.dims = dims
        self.use_lpu = use_lpu
        self.drop_path_rate = drop_path_rate

        print(f"Initializing HistoMamba (SpatialMambaBlock variant)")
        if not MAMBA_AVAILABLE:
            print("  WARNING: Mamba library not available. SpatialMambaBlocks will be SKIPPED.")
        print(f"  Input: {img_size}x{img_size}, Dims: {dims}, Num Classes: {num_classes}")
        print(f"  Use LPU: {use_lpu}, Base DropPath Rate: {drop_path_rate}")

        encoder_layers = nn.ModuleList()
        in_channels = 3
        current_size = img_size
        total_depth = len(dims) if MAMBA_AVAILABLE else 0
        block_idx = 0

        for i, dim in enumerate(dims):
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dim), nn.GELU()
                )
            )
            current_size = math.ceil(current_size / 2)
            if self.use_lpu:
                encoder_layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim))
                encoder_layers.append(nn.BatchNorm2d(dim))
                encoder_layers.append(nn.GELU())
            if Mamba is not None and MAMBA_AVAILABLE:
                current_drop_path = drop_path_rate * (block_idx / (total_depth - 1)) if total_depth > 1 else 0.0
                print(f"  Adding SpatialMambaBlock Stage {i+1} (dim={dim}), DropPath: {current_drop_path:.4f}")
                encoder_layers.append(SpatialMambaBlock(dim=dim, drop_path=current_drop_path, **block_kwargs))
                block_idx += 1
            else:
                print(f"  Info: Mamba not found, SpatialMambaBlock Stage {i+1} skipped.")
            in_channels = dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.final_feature_dim = dims[-1]
        print(f"Encoder output feature dim: {self.final_feature_dim}, approx size: {int(current_size)}x{int(current_size)}")

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LayerNorm(self.final_feature_dim),
            nn.Linear(self.final_feature_dim, num_classes)
        )

        decoder_layers = []
        in_channels_dec = self.final_feature_dim
        for i, target_dim in enumerate(reversed(dims[:-1])):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels_dec, target_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(target_dim), nn.GELU()
                )
            )
            in_channels_dec = target_dim
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels_dec, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)
        print("Decoder configured (used if lambda_recon > 0).")

    def forward(self, x):
        input_H, input_W = x.shape[2:]
        latent = self.encoder(x)
        logits = self.classifier(latent)
        reconstruction = self.decoder(latent)
        rec_H, rec_W = reconstruction.shape[2:]
        if rec_H != input_H or rec_W != input_W:
             reconstruction = nn.functional.interpolate(
                 reconstruction, size=(input_H, input_W), mode='bilinear', align_corners=False
             )
        return {'classification': logits, 'reconstruction': reconstruction}