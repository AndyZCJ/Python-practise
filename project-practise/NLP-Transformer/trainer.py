import torch
import torch.nn as nn
from utils import create_causal_mask, generate_padding_mask


class Trainer:
    def __init__(self, model, device, lr, pad_idx):
        self.model = model.to(device)
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.05)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.pad_idx = pad_idx
        self.device = device
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            src = batch["src_input_ids"].to(self.device)
            tgt = batch["tgt_input_ids"].to(self.device)
            src_mask = generate_padding_mask(src, pad_idx=self.pad_idx)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_padding_mask = generate_padding_mask(tgt_input, pad_idx=self.pad_idx)
            tgt_causal_mask = create_causal_mask(tgt_input.size(1), tgt_input.device)
            tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(1)

            pred = self.model(src, tgt_input, src_mask, tgt_mask)
            loss = self.criterion(pred.reshape(-1, pred.size(-1)), tgt_output.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader, max_batches=None):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            src = batch["src_input_ids"].to(self.device)
            tgt = batch["tgt_input_ids"].to(self.device)
            src_mask = generate_padding_mask(src, pad_idx=self.pad_idx)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_padding_mask = generate_padding_mask(tgt_input, pad_idx=self.pad_idx)
            tgt_causal_mask = create_causal_mask(tgt_input.size(1), tgt_input.device)
            tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(1)

            with torch.no_grad():
                pred = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(pred.reshape(-1, pred.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss/num_batches
        return avg_loss
