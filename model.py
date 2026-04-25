from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class SmallCNN(nn.Module):
    """
    MNISTのような小さいグレースケール画像向けの小型CNN。
    入力: 1 x 28 x 28
    出力: 10クラス
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 28x28 -> pool後 14x14 -> pool後 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 1x28x28 -> 16x14x14
        x = self.pool(F.relu(self.conv2(x)))  # 16x14x14 -> 32x7x7

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    