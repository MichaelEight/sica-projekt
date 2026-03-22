from __future__ import annotations

import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, nb_filters: int = 32, bottleneck_channels: int = 32) -> None:
        super().__init__()
        # Bottleneck before wide convs reduces compute (standard InceptionTime pattern)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.branch_k9 = nn.Conv1d(bottleneck_channels, nb_filters, kernel_size=9, padding="same", bias=False)
        self.branch_k19 = nn.Conv1d(bottleneck_channels, nb_filters, kernel_size=19, padding="same", bias=False)
        self.branch_k39 = nn.Conv1d(bottleneck_channels, nb_filters, kernel_size=39, padding="same", bias=False)

        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False),
        )

        self.norm = nn.BatchNorm1d(nb_filters * 4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        out = torch.cat([self.branch_k9(b), self.branch_k19(b), self.branch_k39(b), self.branch_pool(x)], dim=1)
        return self.act(self.norm(out))


class Inception1DNet(nn.Module):
    """ECG classifier for input (batch, 12, 5000), output raw logits (batch, 8).

    - 6 InceptionBlocks in 2 groups of 3
    - skip connection after every 3 blocks (standard InceptionTime)
    - bottleneck Conv1d(k=1) before wide branches reduces parameters
    - kernel sizes 9/19/39 (odd, symmetric padding) per original InceptionTime paper
    """

    def __init__(self, num_classes: int = 8, dropout: float = 0.5) -> None:
        super().__init__()
        self.stem = nn.Conv1d(12, 32, kernel_size=1, bias=False)

        # Group 1: blocks 1-3 with one skip spanning all three
        self.block1 = InceptionBlock(32, nb_filters=32)
        self.block2 = InceptionBlock(128, nb_filters=32)
        self.block3 = InceptionBlock(128, nb_filters=32)
        self.skip1_proj = nn.Conv1d(32, 128, kernel_size=1, bias=False)
        self.skip1_bn = nn.BatchNorm1d(128)

        # Group 2: blocks 4-6 with one skip spanning all three
        self.block4 = InceptionBlock(128, nb_filters=32)
        self.block5 = InceptionBlock(128, nb_filters=32)
        self.block6 = InceptionBlock(128, nb_filters=32)
        self.skip2_bn = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        # Group 1
        skip1 = self.skip1_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.relu(self.skip1_bn(x + skip1))

        # Group 2
        skip2 = x
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = torch.relu(self.skip2_bn(x + skip2))

        x = torch.mean(x, dim=2)
        x = self.dropout(x)
        return self.head(x)

    @torch.no_grad()
    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        probs = torch.sigmoid(self.forward(x))
        if was_training:
            self.train()
        return probs





