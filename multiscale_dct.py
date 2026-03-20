
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def build_dct_filters(size, device="cpu"):
    filters = []
    for u in range(size):
        for v in range(size):
            filt = np.zeros((size, size))
            for x in range(size):
                for y in range(size):
                    filt[x, y] = (
                        np.cos((2*x+1)*u*np.pi/(2*size)) *
                        np.cos((2*y+1)*v*np.pi/(2*size))
                    )
            cu = np.sqrt(1/size) if u == 0 else np.sqrt(2/size)
            cv = np.sqrt(1/size) if v == 0 else np.sqrt(2/size)
            filters.append(cu * cv * filt)
    filters = np.stack(filters, axis=0)[:, None]
    return torch.tensor(filters, dtype=torch.float32, device=device)


class SingleScaleDCT(nn.Module):
    def __init__(self, size=8, out_channels=64, device="cpu"):
        super().__init__()
        self.size  = size
        n_freq     = size * size
        filters    = build_dct_filters(size, device)
        self.register_buffer("dct_filters", filters)
        self.freq_weight = nn.Parameter(torch.ones(n_freq, 1, 1))
        self.encoder = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        pad      = self.size // 2
        freq_sum = None
        for c in range(C):
            fm       = F.conv2d(x[:, c:c+1], self.dct_filters,
                                stride=1, padding=pad)
            freq_sum = fm if freq_sum is None else freq_sum + fm
        freq     = (freq_sum / C) * self.freq_weight
        freq_3ch = freq[:, :3, :, :]
        return self.encoder(freq_3ch)


class MultiScaleDCT(nn.Module):
    def __init__(self, scales=(8, 16, 32), out_channels=64, device="cpu"):
        super().__init__()
        self.scales       = scales
        self.out_channels = out_channels
        n_scales          = len(scales)

        self.scale_branches = nn.ModuleList([
            SingleScaleDCT(size=s, out_channels=out_channels, device=device)
            for s in scales
        ])

        self.fusion = nn.Sequential(
            nn.Linear(out_channels * n_scales, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_scales),
            nn.Softmax(dim=-1),
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B             = x.shape[0]
        scale_feats   = [branch(x) for branch in self.scale_branches]
        pooled        = [f.mean(dim=[2, 3]) for f in scale_feats]
        pooled_cat    = torch.cat(pooled, dim=1)
        scale_weights = self.fusion(pooled_cat)
        fused = sum(
            scale_weights[:, i].view(B, 1, 1, 1) * scale_feats[i]
            for i in range(len(self.scales))
        )
        return self.output_proj(fused), scale_weights
