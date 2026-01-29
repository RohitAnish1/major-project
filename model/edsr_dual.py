import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class EDSRDual(nn.Module):
    def __init__(self, scale=4, n_feats=64, n_resblocks=16):
        super().__init__()

        # 🔴 ONLY CHANGE IS HERE (6 input channels instead of 3)
        self.head = nn.Conv2d(6, n_feats, 3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(n_feats) for _ in range(n_resblocks)]
        )

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, 3, padding=1)
        )

    def forward(self, lr1, lr2):
        x = torch.cat([lr1, lr2], dim=1)  # [B, 6, H, W]
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x