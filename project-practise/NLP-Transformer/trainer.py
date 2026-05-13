import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, device, lr):
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.05)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        for x, y in train_loader:
            pred