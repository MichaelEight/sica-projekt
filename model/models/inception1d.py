from __future__ import annotations

import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        bottleneck_channels: int = 32,
        kernel_sizes: tuple[int, int, int] = (9, 19, 39),
    ) -> None:
        super().__init__()
        # Bottleneck before wide convs reduces compute (standard InceptionTime pattern)
        k1, k2, k3 = kernel_sizes
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.branch_k9 = nn.Conv1d(bottleneck_channels, nb_filters, kernel_size=k1, padding="same", bias=False)
        self.branch_k19 = nn.Conv1d(bottleneck_channels, nb_filters, kernel_size=k2, padding="same", bias=False)
        self.branch_k39 = nn.Conv1d(bottleneck_channels, nb_filters, kernel_size=k3, padding="same", bias=False)

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

    def __init__(
        self,
        num_classes: int = 8,
        dropout: float = 0.5,
        nb_filters: int = 32,
        kernel_sizes: tuple[int, int, int] = (9, 19, 39),
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(12, nb_filters, kernel_size=1, bias=False)
        channels = nb_filters * 4

        # Group 1: blocks 1-3 with one skip spanning all three
        self.block1 = InceptionBlock(nb_filters, nb_filters=nb_filters, bottleneck_channels=nb_filters, kernel_sizes=kernel_sizes)
        self.block2 = InceptionBlock(channels, nb_filters=nb_filters, bottleneck_channels=nb_filters, kernel_sizes=kernel_sizes)
        self.block3 = InceptionBlock(channels, nb_filters=nb_filters, bottleneck_channels=nb_filters, kernel_sizes=kernel_sizes)
        self.skip1_proj = nn.Conv1d(nb_filters, channels, kernel_size=1, bias=False)
        self.skip1_bn = nn.BatchNorm1d(channels)

        # Group 2: blocks 4-6 with one skip spanning all three
        self.block4 = InceptionBlock(channels, nb_filters=nb_filters, bottleneck_channels=nb_filters, kernel_sizes=kernel_sizes)
        self.block5 = InceptionBlock(channels, nb_filters=nb_filters, bottleneck_channels=nb_filters, kernel_sizes=kernel_sizes)
        self.block6 = InceptionBlock(channels, nb_filters=nb_filters, bottleneck_channels=nb_filters, kernel_sizes=kernel_sizes)
        self.skip2_bn = nn.BatchNorm1d(channels)

        # Concat-pooling (mean+max) doubles channels (2 * 4*nb_filters), so use a
        # small head like in PTB-XL benchmark: 256->128->BN->DO(0.25)->ReLU->DO(0.5)->num_classes
        self.head = nn.Sequential(
            nn.Linear(channels * 2, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.25),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

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

        # concat-pooling: concatenate mean and max along temporal dim (dim=2)
        x_mean = torch.mean(x, dim=2)
        x_max = torch.max(x, dim=2)[0]
        x = torch.cat([x_mean, x_max], dim=1)
        return self.head(x)

    @torch.no_grad()
    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        probs = torch.sigmoid(self.forward(x))
        if was_training:
            self.train()
        return probs





