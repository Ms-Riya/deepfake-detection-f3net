
import torch
import torch.nn as nn
import torch.nn.functional as F
from xception import load_xception, Xception
from multiscale_dct import MultiScaleDCT, build_dct_filters
from compression_aug import CompressionEmbedding, DEFAULT_COMPRESSION_IDX


class LFSBranch(nn.Module):
    """Local Frequency Statistics — unchanged from baseline."""
    def __init__(self, size=8, out_channels=32, device="cpu"):
        super().__init__()
        self.size = size
        filters = build_dct_filters(size, device)
        self.register_buffer("dct_filters", filters)
        self.lfs_net = nn.Sequential(
            nn.Conv2d(size * size, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        pad  = self.size // 2
        local_freq = F.conv2d(gray, self.dct_filters, stride=1, padding=pad)
        return self.lfs_net(local_freq)


class MixBlock(nn.Module):
    """Cross-attention fusion — unchanged from baseline."""
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
        fad = self.proj_fad(fad_feat)
        lfs = self.proj_lfs(lfs_feat)
        if fad.shape[2:] != lfs.shape[2:]:
            lfs = F.interpolate(lfs, size=fad.shape[2:],
                                mode="bilinear", align_corners=False)
        attn = self.attn(torch.cat([fad, lfs], dim=1))
        return self.act(self.bn(fad * attn + lfs * (1 - attn)))


class XceptionWithFreqInput(nn.Module):
    """
    Modified Xception that accepts an additional frequency feature map
    and adds it to the early spatial features.

    Why this is better than projecting 128→3:
        Instead of squeezing 128 channels into 3 and losing information,
        we let Xception process the image normally for 2 blocks,
        then ADD the multi-scale frequency features at that point.
        This way:
          - Xception keeps its full spatial understanding
          - Frequency features are injected where spatial resolution
            is still high enough to be useful (after block2 = 256 channels)
          - No information bottleneck
    """
    def __init__(self, pretrained_path, freq_channels=128):
        super().__init__()

        # Load full pretrained Xception
        base = load_xception(pretrained_path, num_classes=2)

        # Split into early and late parts
        # Early: up to block2 output (256 channels, ~75x75 spatial)
        # Late:  block3 onwards to 2048 channels
        self.early = nn.Sequential(
            base.conv1, base.bn1, base.relu,
            base.conv2, base.bn2, base.relu,
            base.block1,
            base.block2,
        )

        self.late = nn.Sequential(
            base.block3,
            base.block4,  base.block5,  base.block6,
            base.block7,  base.block8,  base.block9,
            base.block10, base.block11, base.block12,
            base.conv3,   base.bn3,     base.relu,
            base.conv4,   base.bn4,     base.relu,
        )

        # Project frequency features to match early Xception output (256 ch)
        # This is a MUCH smaller bottleneck: 128→256 instead of 128→3
        self.freq_inject = nn.Sequential(
            nn.Conv2d(freq_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, freq_feat):
        """
        Args:
            x        : (B, 3, 299, 299) — original image
            freq_feat: (B, 128, H, W)   — multi-scale DCT features

        Returns:
            (B, 2048, h, w) — deep features with frequency injected
        """
        # Process image through early Xception blocks
        spatial = self.early(x)                    # (B, 256, ~75, ~75)

        # Resize frequency features to match spatial size
        freq = F.interpolate(freq_feat, size=spatial.shape[2:],
                             mode="bilinear", align_corners=False)
        freq = self.freq_inject(freq)              # (B, 256, ~75, ~75)

        # Add frequency features to spatial features
        # Addition (not concat) keeps channel count the same
        # and forces the model to learn complementary representations
        combined = spatial + freq                  # (B, 256, ~75, ~75)

        # Process through rest of Xception
        return self.late(combined)                 # (B, 2048, h, w)


class F3NetNovel(nn.Module):
    """
    F3Net + Novelty A (Multi-Scale DCT) + Novelty B (Compression Embedding)

    Architecture fix vs previous version:
        Multi-scale features are INJECTED into Xception at block2 output
        instead of being projected 128→3 at the input.
        This preserves information and avoids the bottleneck.

    Args:
        pretrained_path   : path to xception-b5690688.pth
        device            : "cpu" or "cuda"
        use_multiscale    : True = Novelty A, False = baseline single-scale
        use_comp_embedding: True = Novelty B, False = no compression info
    """
    def __init__(self,
                 pretrained_path,
                 device="cpu",
                 use_multiscale=True,
                 use_comp_embedding=True):
        super().__init__()
        self.device             = device
        self.use_multiscale     = use_multiscale
        self.use_comp_embedding = use_comp_embedding

        # ── FAD Branch ────────────────────────────────────────────────────
        if use_multiscale:
            # NOVELTY A: 3 DCT scales with adaptive fusion
            self.fad = MultiScaleDCT(
                scales=(8, 16, 32),
                out_channels=64,
                device=device
            )
            # Modified Xception: injects 128-ch freq features at block2
            self.fad_backbone = XceptionWithFreqInput(
                pretrained_path, freq_channels=128
            )
        else:
            # Baseline: single-scale FAD + standard Xception
            from f3net import FADBranch
            self.fad          = FADBranch(size=8, device=device)
            self.fad_backbone = load_xception(pretrained_path, num_classes=2)

        # ── LFS Branch (unchanged from baseline) ──────────────────────────
        self.lfs          = LFSBranch(size=8, out_channels=32, device=device)
        self.lfs_backbone = load_xception(pretrained_path, num_classes=2)

        # ── MixBlock (unchanged) ──────────────────────────────────────────
        self.mixblock = MixBlock(2048, 32, out_channels=512)

        # ── NOVELTY B: Compression Embedding ──────────────────────────────
        comp_embed_dim = 64 if use_comp_embedding else 0
        if use_comp_embedding:
            self.comp_embed = CompressionEmbedding(
                n_levels=3, embed_dim=comp_embed_dim
            )

        # ── Classifier ────────────────────────────────────────────────────
        classifier_in = 512 + comp_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, compression_level=None):
        B = x.shape[0]

        # ── FAD branch ────────────────────────────────────────────────────
        scale_weights = None

        if self.use_multiscale:
            freq_feat, scale_weights = self.fad(x)      # (B, 128, H, W)
            # Inject frequency features into Xception at block2
            fad_feat = self.fad_backbone(x, freq_feat)  # (B, 2048, h, w)
        else:
            fad_input = self.fad(x)                     # (B, 3, H, W)
            fad_feat  = self.fad_backbone.features(fad_input)

        # ── LFS branch ────────────────────────────────────────────────────
        lfs_feat = self.lfs(x)                          # (B, 32, H, W)

        # ── MixBlock fusion ───────────────────────────────────────────────
        mixed = self.mixblock(fad_feat, lfs_feat)       # (B, 512, h, w)

        # ── Global pooling ────────────────────────────────────────────────
        features = self.pool(mixed).flatten(1)          # (B, 512)

        # ── NOVELTY B: compression embedding ──────────────────────────────
        if self.use_comp_embedding:
            if compression_level is None:
                compression_level = torch.ones(
                    B, dtype=torch.long, device=x.device
                ) * DEFAULT_COMPRESSION_IDX
            comp_vec = self.comp_embed(compression_level)
            features = torch.cat([features, comp_vec], dim=1)

        # ── Classify ──────────────────────────────────────────────────────
        logit = self.classifier(features).squeeze(-1)
        return logit, scale_weights
