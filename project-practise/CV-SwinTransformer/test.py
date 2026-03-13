from attention import SwinBlock
import torch

x = torch.randn(16, 48*48, 128)

block = SwinBlock(128, (48, 48),8,window_size=6,shift_size=3)

print(block(x).shape)