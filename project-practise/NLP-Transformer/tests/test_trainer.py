import unittest

import torch
import torch.nn as nn

from trainer import Trainer


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        self.calls += 1
        logits = torch.zeros(tgt.size(0), tgt.size(1), 8)
        logits[:, :, 1] = self.weight
        return logits


class TrainerTests(unittest.TestCase):
    def test_evaluate_stops_after_max_batches(self):
        model = CountingModel()
        trainer = Trainer(model, device="cpu", lr=0.001, pad_idx=0)
        batches = [
            {
                "src_input_ids": torch.tensor([[1, 2, 0]]),
                "tgt_input_ids": torch.tensor([[1, 2, 0]]),
            },
            {
                "src_input_ids": torch.tensor([[1, 3, 0]]),
                "tgt_input_ids": torch.tensor([[1, 3, 0]]),
            },
        ]

        trainer.evaluate(batches, max_batches=1)

        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
