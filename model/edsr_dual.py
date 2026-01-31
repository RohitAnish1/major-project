import torch
import torch.nn as nn

# ------------------------
# MeanShift (official EDSR)
# ------------------------
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        rgb_mean = (0.4488, 0.4371, 0.4040)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)

        self.weight.requires_grad = False
        self.bias.requires_grad = False


# ------------------------
# Residual Block (EDSR)
# ------------------------
class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale


# ------------------------
# EDSR Dual-LR Model
# ------------------------
class EDSRDual(nn.Module):
    def __init__(self, scale=4, n_feats=64, n_resblocks=16, rgb_range=255):
        super().__init__()

        # Mean normalization (same as EDSR)
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # Dual-LR input → 6 channels
        self.head = nn.Conv2d(6, n_feats, 3, padding=1)

        # Deep residual trunk
        self.body = nn.Sequential(
            *[ResBlock(n_feats) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )

        # Upsampling tail (PixelShuffle)
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, 3, padding=1)
        )

    def forward(self, lr1, lr2):
        """
        lr1, lr2: [B, 3, H, W] aligned low-resolution inputs
        returns:  [B, 3, sH, sW] super-resolved output
        """

        # Normalize BOTH inputs (critical fix)
        lr1 = self.sub_mean(lr1)
        lr2 = self.sub_mean(lr2)

        # Fuse dual inputs
        x = torch.cat([lr1, lr2], dim=1)   # [B, 6, H, W]
        x = self.head(x)

        # Global residual learning
        res = self.body(x)
        res = res + x                      # EDSR global skip

        # Upsample and reconstruct
        x = self.tail(res)

        # De-normalize
        x = self.add_mean(x)
        return x
