"""
Inception1D — architektura sieci neuronowej do klasyfikacji sygnałów EKG.
Bazuje na benchmarku PTB-XL (Strodthoff et al., 2021).
"""
import torch
import torch.nn as nn


class InceptionBlock1d(nn.Module):
    """Single Inception block for 1D signals."""

    def __init__(self, in_channels, nb_filters=32, bottleneck_size=32, kernel_sizes=(39, 19, 9)):
        super().__init__()

        # Bottleneck: 1x1 convolution to reduce channels
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)

        # Parallel conv branches with different kernel sizes
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.convs.append(
                nn.Conv1d(bottleneck_size, nb_filters, kernel_size=ks, padding=ks // 2, bias=False)
            )

        # MaxPool branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        # Output: 4 branches * nb_filters
        out_channels = nb_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Bottleneck
        x_bottleneck = self.bottleneck(x)

        # Conv branches
        outputs = [conv(x_bottleneck) for conv in self.convs]

        # MaxPool branch
        x_pool = self.conv_pool(self.maxpool(x))
        outputs.append(x_pool)

        # Concatenate and activate
        out = torch.cat(outputs, dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ShortcutBlock1d(nn.Module):
    """Residual shortcut connection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, residual):
        shortcut = self.bn(self.conv(residual))
        return self.relu(x + shortcut)


class InceptionBackbone(nn.Module):
    """Stack of Inception blocks with residual shortcuts."""

    def __init__(self, input_channels=12, nb_filters=32, depth=6):
        super().__init__()

        block_channels = nb_filters * 4  # 4 branches per block

        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        for i in range(depth):
            in_ch = input_channels if i == 0 else block_channels
            self.blocks.append(InceptionBlock1d(in_ch, nb_filters))

            # Residual shortcut every 3 blocks
            if (i + 1) % 3 == 0:
                shortcut_in = input_channels if i < 3 else block_channels
                self.shortcuts.append(ShortcutBlock1d(shortcut_in, block_channels))
            else:
                self.shortcuts.append(None)

        self.output_channels = block_channels

    def forward(self, x):
        residual = x

        for i, (block, shortcut) in enumerate(zip(self.blocks, self.shortcuts)):
            x = block(x)
            if shortcut is not None:
                x = shortcut(x, residual)
                residual = x

        return x


class Inception1d(nn.Module):
    """
    Inception1D model for multi-label ECG classification.

    Input: (batch, 12, 5000) — 12-lead ECG, 10s @ 500Hz
    Output: (batch, num_classes) — logits
    """

    def __init__(self, input_channels=12, num_classes=8, nb_filters=32, depth=6, dropout=0.5):
        super().__init__()

        self.backbone = InceptionBackbone(input_channels, nb_filters, depth)

        backbone_out = self.backbone.output_channels  # 128

        # Adaptive pooling: avg + max concatenated
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(backbone_out * 2),
            nn.Dropout(dropout),
            nn.Linear(backbone_out * 2, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)

        avg = self.avg_pool(features)
        mx = self.max_pool(features)
        pooled = torch.cat([avg, mx], dim=1)

        logits = self.head(pooled)
        return logits

    def get_feature_maps(self, x):
        """Return backbone feature maps (for Grad-CAM)."""
        return self.backbone(x)


def build_model(input_channels=12, num_classes=8, **kwargs):
    """Build and return an Inception1D model."""
    return Inception1d(input_channels=input_channels, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Quick test
    model = build_model()
    dummy = torch.randn(2, 12, 5000)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
