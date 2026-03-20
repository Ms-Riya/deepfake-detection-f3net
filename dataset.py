import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

DATASET_ROOT = "/home/pesu/deepfake_project/dataset"
IMAGE_SIZE   = 299
BATCH_SIZE   = 16
NUM_WORKERS  = 4

def get_train_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def get_val_transform():
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

class FFppDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        self.samples   = []

        split_dir = os.path.join(DATASET_ROOT, split)

        # Real samples — label 0
        real_dir = os.path.join(split_dir, "real")
        if os.path.exists(real_dir):
            for vid_id in sorted(os.listdir(real_dir)):
                vid_path = os.path.join(real_dir, vid_id)
                if not os.path.isdir(vid_path):
                    continue
                for frame in sorted(os.listdir(vid_path)):
                    if frame.endswith('.jpg'):
                        self.samples.append((os.path.join(vid_path, frame), 0))

        # Fake samples — label 1
        fake_dir = os.path.join(split_dir, "fake")
        if os.path.exists(fake_dir):
            for method in sorted(os.listdir(fake_dir)):
                method_path = os.path.join(fake_dir, method)
                if not os.path.isdir(method_path):
                    continue
                for vid_id in sorted(os.listdir(method_path)):
                    vid_path = os.path.join(method_path, vid_id)
                    if not os.path.isdir(vid_path):
                        continue
                    for frame in sorted(os.listdir(vid_path)):
                        if frame.endswith('.jpg'):
                            self.samples.append((os.path.join(vid_path, frame), 1))

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"[{split:5s}] {len(self.samples):6d} samples | Real: {n_real:5d} | Fake: {n_fake:5d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

    def get_sample_weights(self):
        labels  = [l for _, l in self.samples]
        n_real  = labels.count(0)
        n_fake  = labels.count(1)
        n_total = len(labels)
        w_real  = n_total / (2.0 * n_real)
        w_fake  = n_total / (2.0 * n_fake)
        return torch.tensor([w_real if l == 0 else w_fake for l in labels], dtype=torch.double)

def get_dataloaders():
    train_ds = FFppDataset("train", transform=get_train_transform())
    val_ds   = FFppDataset("val",   transform=get_val_transform())
    test_ds  = FFppDataset("test",  transform=get_val_transform())

    weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return {"train": train_loader, "val": val_loader, "test": test_loader}
