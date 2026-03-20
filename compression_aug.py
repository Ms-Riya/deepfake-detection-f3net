
import io
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T


# ── Novelty B Part 1 — Compression Augmentation ──────────────────────────────

class JPEGCompressAug:
    """
    Randomly apply JPEG compression during training.

    Why this helps:
        F3Net trained on c23 drops on c40 because heavy compression
        destroys the high-frequency artifacts it learned to detect.
        By simulating compression during training, the model learns
        artifacts that SURVIVE compression — making it robust to c40.

    What it does:
        With probability `prob`, compress the image to JPEG at a
        random quality between quality_low and quality_high,
        then decompress back to PIL Image.
        The round-trip compression introduces the same artifacts
        that real video compression would introduce.

    Args:
        prob         : probability of applying compression (0.5 = 50% of batches)
        quality_low  : minimum JPEG quality (lower = more compression = more artifacts)
        quality_high : maximum JPEG quality (higher = less compression)
    """
    def __init__(self, prob=0.5, quality_low=65, quality_high=95):
        self.prob         = prob
        self.quality_low  = quality_low
        self.quality_high = quality_high

    def __call__(self, img):
        """
        Args:
            img: PIL Image

        Returns:
            PIL Image — possibly JPEG compressed
        """
        if random.random() > self.prob:
            return img   # No augmentation this time

        quality = random.randint(self.quality_low, self.quality_high)

        # Compress to JPEG in memory buffer then read back
        # This is the key step — it introduces real JPEG artifacts
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).copy()
        return compressed


def get_train_transform_with_compression():
    """
    Training transform that includes compression augmentation.
    Drop-in replacement for get_train_transform() in dataset.py
    """
    return T.Compose([
        T.Resize((299, 299)),
        JPEGCompressAug(prob=0.5, quality_low=65, quality_high=95),  # NEW
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ── Novelty B Part 2 — Compression Level Embedding ───────────────────────────

class CompressionEmbedding(nn.Module):
    """
    Tells the model what compression level it is dealing with.

    Why this helps:
        At inference time, if we know the compression level, we can
        tell the model to adjust its sensitivity.
        "This is heavily compressed — don't rely on high-frequency artifacts"
        "This is raw — high-frequency artifacts are reliable signals"

    How it works:
        compression_level is an integer: 0 = c0, 1 = c23, 2 = c40
        We map it to a learned embedding vector (embed_dim dimensions)
        This vector gets added to the classifier input features.

    Args:
        n_levels  : number of compression levels (3 for c0/c23/c40)
        embed_dim : size of embedding vector — must match classifier input
    """
    def __init__(self, n_levels=3, embed_dim=512):
        super().__init__()
        # Simple lookup table: each compression level gets its own vector
        self.embedding = nn.Embedding(n_levels, embed_dim)

        # Initialize to small values so it starts as a minor correction
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, compression_level):
        """
        Args:
            compression_level: (B,) LongTensor — 0, 1, or 2 per sample

        Returns:
            (B, embed_dim) — embedding vector to add to features
        """
        return self.embedding(compression_level)


# ── Compression level map ─────────────────────────────────────────────────────
# Use this when building dataloaders to assign integer labels
# to compression levels

COMPRESSION_TO_IDX = {
    "c0":  0,
    "c23": 1,
    "c40": 2,
}

# Default during training — we trained on c23
DEFAULT_COMPRESSION_IDX = 1
