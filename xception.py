
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1     = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.conv1(x))


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1,
                 start_with_relu=True, grow_first=True):
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip   = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep       = []
        filters   = in_filters

        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x    = self.rep(inp)
        skip = self.skipbn(self.skip(inp)) if self.skip is not None else inp
        return x + skip


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        self.block1  = Block(64,  128,  2, 2, start_with_relu=False, grow_first=True)
        self.block2  = Block(128, 256,  2, 2, start_with_relu=True,  grow_first=True)
        self.block3  = Block(256, 728,  2, 2, start_with_relu=True,  grow_first=True)
        self.block4  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block5  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block6  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block7  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block8  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block9  = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block10 = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block11 = Block(728, 728,  3, 1, start_with_relu=True,  grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True,  grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3   = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4   = nn.BatchNorm2d(2048)
        self.fc    = nn.Linear(2048, num_classes)

    def features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x);  x = self.block2(x);  x = self.block3(x)
        x = self.block4(x);  x = self.block5(x);  x = self.block6(x)
        x = self.block7(x);  x = self.block8(x);  x = self.block9(x)
        x = self.block10(x); x = self.block11(x); x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        return x

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.fc(x)


def load_xception(pretrained_path, num_classes=2):
    model      = Xception(num_classes=1000)
    pretrained = torch.load(pretrained_path, map_location="cpu")

    # ── Fix shape mismatch ────────────────────────────────────────────────
    # The saved checkpoint stores pointwise conv weights as 2D (out, in)
    # but nn.Conv2d expects 4D (out, in, 1, 1).
    # We fix this by unsqueezing the last two dimensions.
    fixed = {}
    for k, v in pretrained.items():
        if "pointwise.weight" in k and v.dim() == 2:
            v = v.unsqueeze(-1).unsqueeze(-1)   # (out, in) → (out, in, 1, 1)
        fixed[k] = v

    model.load_state_dict(fixed, strict=False)
    print("Xception pretrained weights loaded ✓")

    # Replace final FC for binary classification
    model.fc = nn.Linear(2048, num_classes)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)

    return model
