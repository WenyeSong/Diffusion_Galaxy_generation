"""
Lightweight ResNet-style CNN regressor for 4 galaxy morphology parameters.
Input:  (B, 5, 64, 64)  float32
Output: (B, 4)           float32  (normalised targets)
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class MorphCNN(nn.Module):
    """
    Encoder: 5 conv stages that halve spatial dims each time
             64x64 -> 32 -> 16 -> 8 -> 4 -> 2
    Head:    global avg pool -> FC -> 4 outputs
    """

    def __init__(self, n_outputs: int = 4):
        super().__init__()

        def conv_stage(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                ResBlock(out_ch),
            )

        self.encoder = nn.Sequential(
            conv_stage(5,   32),   # -> (B, 32, 32, 32)
            conv_stage(32,  64),   # -> (B, 64, 16, 16)
            conv_stage(64, 128),   # -> (B,128,  8,  8)
            conv_stage(128, 256),  # -> (B,256,  4,  4)
            conv_stage(256, 256),  # -> (B,256,  2,  2)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # -> (B, 256, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x):
        return self.head(self.pool(self.encoder(x)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MorphCNN()
    x = torch.randn(4, 5, 64, 64)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Trainable parameters: {count_parameters(model):,}")
