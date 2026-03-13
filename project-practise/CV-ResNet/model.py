import torch
import torch.nn as nn

class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Flatten → Linear(784 → 128) → ReLU → Linear(128 → 10)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)
