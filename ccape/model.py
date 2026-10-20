"""U-Net with spatial dropout in the bottleneck + decoder, so it doubles as an MC-Dropout sampler."""
import torch
from torch import nn


def block(ci, co, p):
    layers = [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
              nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
    if p > 0:
        layers.append(nn.Dropout2d(p))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, c=32, depth=4, p=0.2, cin=3):
        super().__init__()
        ch = [c << i for i in range(depth + 1)]
        self.enc = nn.ModuleList([block(cin, ch[0], 0)] + [block(ch[i], ch[i + 1], p if i == depth - 1 else 0) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([block(2 * ch[i], ch[i], p) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)[:, 0]

    def mc_mode(self):
        """Eval mode (frozen BatchNorm) but dropout stays stochastic."""
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()
        return self
