import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
class Trainer:
    def __init__(self, model, device, lr):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.history = {
            "train_loss": [],
            "val_acc": []
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss/len(train_loader)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def eval_epoch(self, test_loader):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                total += y.size(0)
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                label = torch.argmax(pred, dim=1)
                correct += (label==y).sum().item()
        val_acc = correct/total
        self.history["val_acc"].append(val_acc)

        return val_acc


