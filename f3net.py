
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xception import load_xception


def build_dct_filters(size=8, channel=1, device="cpu"):
    n = size
    dct_basis = np.zeros((n * n, n, n), dtype=np.float32)
    for u in range(n):
        for v in range(n):
            for x in range(n):
                for y in range(n):
                    cu = (1.0 / np.sqrt(2)) if u == 0 else 1.0
                    cv = (1.0 / np.sqrt(2)) if v == 0 else 1.0
                    dct_basis[u * n + v, x, y] = (
                        (2.0 / n) * cu * cv *
                        np.cos((2*x + 1) * u * np.pi / (2*n)) *
                        np.cos((2*y + 1) * v * np.pi / (2*n))
                    )
    # Shape: (n^2, 1, n, n) — one filter per frequency, single channel
    dct_basis = dct_basis[:, np.newaxis, :, :]
    return torch.from_numpy(dct_basis).to(device)


class FADBranch(nn.Module):
    """
    Frequency-Aware Decomposition branch.

    How it works:
      1. Apply DCT filters SEPARATELY to each RGB channel
      2. Get 64 frequency components per channel = 192 total
      3. Learn weights for each frequency component
      4. Reconstruct a forgery-aware 3-channel feature map
    """
    def __init__(self, size=8, device="cpu"):
        super().__init__()
        self.size = size

        # DCT filters: shape (64, 1, 8, 8) — applied per channel
        dct_filters = build_dct_filters(size=size, channel=1, device=device)
        self.register_buffer("dct_filters", dct_filters)

        # Learnable weight for each of the 64 frequency components
        # One weight per frequency, shared across RGB channels
        self.freq_weight = nn.Parameter(torch.ones(size * size, 1, 1))

        # Reconstruct: take weighted frequency sum back to 3-channel spatial map
        self.reconstruct = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape        # C = 3
        size = self.size

        # Apply DCT filters to each channel independently
        # Process each channel separately then stack
        channel_outputs = []
        for c in range(C):
            # x[:, c:c+1] shape: (B, 1, H, W)
            # dct_filters shape: (64, 1, 8, 8)
            # output shape: (B, 64, H//8, W//8)
            freq_c = F.conv2d(x[:, c:c+1], self.dct_filters,
                              stride=size, padding=0)
            channel_outputs.append(freq_c)

        # Stack: (B, 3, 64, H//8, W//8)
        freq = torch.stack(channel_outputs, dim=1)

        # Apply learnable frequency weights: (64, 1, 1) broadcasts over spatial dims
        # freq_weight selects which DCT frequencies matter for forgery detection
        weighted = freq * self.freq_weight.unsqueeze(0).unsqueeze(0)  # (B, 3, 64, H', W')

        # Sum across frequency components → (B, 3, H', W')
        freq_sum = weighted.sum(dim=2)

        # Upsample back to original spatial size
        freq_up = F.interpolate(freq_sum, size=(H, W), mode="bilinear", align_corners=False)

        # Reconstruct spatial features from frequency info
        return self.reconstruct(freq_up)


class LFSBranch(nn.Module):
    """
    Local Frequency Statistics branch.
    Slides DCT filters across image to capture local frequency patterns.
    Detects local inconsistencies at blending boundaries.
    """
    def __init__(self, size=8, out_channels=32, device="cpu"):
        super().__init__()
        self.size = size

        # DCT filters: (64, 1, 8, 8)
        dct_filters = build_dct_filters(size=size, channel=1, device=device)
        self.register_buffer("dct_filters", dct_filters)

        # Map 64 local frequency components to feature channels
        self.lfs_net = nn.Sequential(
            nn.Conv2d(size * size, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Convert to grayscale — forgery artifacts are luminance-based
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Slide DCT filters across image with padding to preserve spatial size
        pad        = self.size // 2
        local_freq = F.conv2d(gray, self.dct_filters, stride=1, padding=pad)
        # local_freq: (B, 64, H, W) — local frequency at every position

        return self.lfs_net(local_freq)  # (B, out_channels, H, W)


class MixBlock(nn.Module):
    """
    Cross-attention fusion between FAD and LFS.
    Learns which local frequency patterns (LFS) are important
    given the global frequency context (FAD).
    """
    def __init__(self, in_channels_fad, in_channels_lfs, out_channels=512):
        super().__init__()
        self.proj_fad = nn.Conv2d(in_channels_fad, out_channels, 1, bias=False)
        self.proj_lfs = nn.Conv2d(in_channels_lfs, out_channels, 1, bias=False)
        self.attn = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, fad_feat, lfs_feat):
        fad = self.proj_fad(fad_feat)   # (B, out_channels, H_fad, W_fad)
        lfs = self.proj_lfs(lfs_feat)   # (B, out_channels, H_lfs, W_lfs)

        # Resize lfs to match fad spatial dimensions
        if fad.shape[2:] != lfs.shape[2:]:
            lfs = F.interpolate(lfs, size=fad.shape[2:],
                                mode="bilinear", align_corners=False)

        # Attention gate: which features matter?
        attn = self.attn(torch.cat([fad, lfs], dim=1))  # (B, out_channels, H, W)

        # Weighted blend of FAD and LFS
        return self.act(self.bn(fad * attn + lfs * (1 - attn)))


class F3Net(nn.Module):
    """
    Full F3Net — Frequency in Face Forgery Network (ECCV 2020)
    mode: "FAD", "LFS", "Both", or "Mix" (full model, recommended)
    """
    def __init__(self, pretrained_path, mode="Mix", device="cpu"):
        super().__init__()
        assert mode in ["FAD", "LFS", "Both", "Mix"]
        self.mode   = mode
        self.device = device

        if mode in ["FAD", "Both", "Mix"]:
            self.fad          = FADBranch(size=8, device=device)
            self.fad_backbone = load_xception(pretrained_path, num_classes=2)

        if mode in ["LFS", "Both", "Mix"]:
            self.lfs          = LFSBranch(size=8, out_channels=32, device=device)
            self.lfs_backbone = load_xception(pretrained_path, num_classes=2)

        if mode == "Mix":
            self.mixblock = MixBlock(2048, 32, out_channels=512)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
            )

        elif mode == "Both":
            self.classifier = nn.Sequential(
                nn.Linear(4096, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 1),
            )

    def forward(self, x):
        if self.mode == "FAD":
            feat  = self.fad_backbone.features(self.fad(x))
            feat  = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            logit = self.fad_backbone.fc(feat)
            return logit[:, 1] - logit[:, 0]

        elif self.mode == "LFS":
            lfs_up = F.interpolate(self.lfs(x), size=(299, 299),
                                   mode="bilinear", align_corners=False)
            feat   = self.lfs_backbone.features(lfs_up)
            feat   = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            logit  = self.lfs_backbone.fc(feat)
            return logit[:, 1] - logit[:, 0]

        elif self.mode == "Both":
            fad_feat = F.adaptive_avg_pool2d(
                self.fad_backbone.features(self.fad(x)), 1).flatten(1)
            lfs_up   = F.interpolate(self.lfs(x), size=(299, 299),
                                     mode="bilinear", align_corners=False)
            lfs_feat = F.adaptive_avg_pool2d(
                self.lfs_backbone.features(lfs_up), 1).flatten(1)
            return self.classifier(torch.cat([fad_feat, lfs_feat], dim=1)).squeeze(-1)

        elif self.mode == "Mix":
            fad_feat = self.fad_backbone.features(self.fad(x))  # (B, 2048, H, W)
            lfs_feat = self.lfs(x)                               # (B, 32, H, W)
            mixed    = self.mixblock(fad_feat, lfs_feat)         # (B, 512, H, W)
            return self.classifier(mixed).squeeze(-1)            # (B,)
