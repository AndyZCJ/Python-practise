import unittest

import torch

from inference import greedy_decode


class FakeModel:
    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, src, tgt, src_mask=None, tgt_mask=None):
        self.calls += 1
        logits = torch.zeros(tgt.size(0), tgt.size(1), 120)
        next_token = 7 if self.calls == 1 else 102
        logits[:, -1, next_token] = 10.0
        return logits


class GreedyDecodeTests(unittest.TestCase):
    def test_greedy_decode_stops_after_sep_token(self):
        model = FakeModel()
        src = torch.tensor([[101, 11, 12, 102, 0]])

        result = greedy_decode(
            model=model,
            src=src,
            pad_token_id=0,
            bos_token_id=101,
            eos_token_id=102,
            max_length=5,
            device=torch.device("cpu"),
        )

        self.assertEqual(result, [101, 7, 102])


if __name__ == "__main__":
    unittest.main()
