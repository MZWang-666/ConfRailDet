import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainHead(nn.Module):
    def __init__(self, in_channels=1024, embed_dim=256):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 256, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # domain classifier (2类：RGB vs B显)
        self.cls = nn.Linear(embed_dim, 2)

    def forward(self, x):
        # x: S5 feature [B,C,H,W]

        x = self.conv(x)
        x = self.pool(x).flatten(1)

        embedding = self.mlp(x)
        logits = self.cls(embedding)

        return embedding, logits