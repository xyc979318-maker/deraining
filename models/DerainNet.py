import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class DerainNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_blocks=8):
        super(DerainNet, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.branch_full = self.Blocks(base_channels, num_blocks)
        self.branch_half = self.Blocks(base_channels, num_blocks)
        self.branch_quarter = self.Blocks(base_channels, num_blocks)

        self.pool_half = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_quarter = nn.MaxPool2d(kernel_size=4, stride=4)

        self.upsample_half = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_quarter = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.trans= nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        )

    def Blocks(self, channels, num_blocks):   #将前面我们写的8层残差块封装在一个函数中
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_initial = self.initial_conv(x)

        branch1_out = self.branch_full(x_initial) + x_initial

        x_half = self.pool_half(x_initial)
        branch2_out = self.branch_half(x_half) + x_half
        branch2_out = self.upsample_half(branch2_out)

        x_quarter = self.pool_quarter(x_initial)
        branch3_out = self.branch_quarter(x_quarter) + x_quarter
        branch3_out = self.upsample_quarter(branch3_out)

        concatenated = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        rain_layer = self.trans(concatenated)
        deraining = x - rain_layer
        return deraining