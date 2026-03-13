import torch
import argparse
import matplotlib.pyplot as plt
from model import SimpleNet
from ResNet import MiniResNet
from dataset import get_dataloaders
import os
from trainer import Trainer
import json
from torchvision import models
import torch.nn as nn

def main():
    os.makedirs("outputs", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, args.lr)

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    #model = models.resnet18(weights="IMAGENET1K_V1") #加载权重
    #for p in model.parameters(): #冻结backbone
    #    p.requires_grad = False
    #model.fc = nn.Linear(model.fc.in_features, 10)
    model = MiniResNet()
    model_trainer = Trainer(model, device, args.lr)
    epochs = args.epochs

    best_acc = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model_trainer.train_epoch(train_loader)

        acc = model_trainer.eval_epoch(test_loader)
        if acc > best_acc:
            best_acc = acc
            print("Best acc:", best_acc)
            torch.save(model_trainer.model.state_dict(),"./outputs/best.pt")

    losses = model_trainer.history["train_loss"]
    accs = model_trainer.history["val_acc"]
    with open("./outputs/train_history.json","w") as f:
        json.dump(model_trainer.history, f)
    plt.figure()
    plt.plot(losses)
    plt.title("Train losses")
    plt.savefig("./outputs/train_loss.png")

    plt.figure()
    plt.plot(accs)
    plt.title("Validation accuracy")
    plt.savefig("./outputs/val_acc.png")
if __name__ == "__main__":
    main()
