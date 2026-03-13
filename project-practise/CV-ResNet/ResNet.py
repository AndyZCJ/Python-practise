import torch
import torch.nn as nn
from numpy.ma.core import identity


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        '''
        Conv2D: H_out = (H+2P-K)/S +1
        '''
        # TODO
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride!=1 or in_ch!=out_ch:
            self.downsample  = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.downsample  = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # TODO
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO:
        # stem conv
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # TODO:
        # layer1 = 2 blocks (16→16)
        self.layer1 = nn.ModuleList([
            BasicBlock(16, 16),
            BasicBlock(16, 16),
        ])
        # TODO:
        # layer2 = 2 blocks (16→32, stride=2)
        self.layer2 = nn.ModuleList([
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32),
        ])
        # TODO:
        # layer3 = 2 blocks (32→64, stride=2)
        self.layer3 = nn.ModuleList([
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64),
        ])
        # TODO:
        # global avg pool
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 10)
        # fc

    def forward(self, x):
        # TODO
        x = self.conv1(x)
        for block in self.layer1:
            x = block(x)

        for block in self.layer2:
            x = block(x)

        for block in self.layer3:
            x = block(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)