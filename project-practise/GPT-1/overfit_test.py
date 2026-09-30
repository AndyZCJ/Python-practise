import torch
from gpt import config, GPT, gpt_loss
from utils import setup_seed

setup_seed(1)
test_config = config(vocab_size=256, context_length=16, embedding_dim=64, num_heads=4,
                      num_layers=2, ffn_dim=256, dropout=0, layer_norm_eps=1e-5, use_bias=True)   

test_gpt = GPT(config=test_config)
optimizer = torch.optim.AdamW(test_gpt.parameters(), lr=1e-3)
data = torch.randint(0, 256, (4, 17))

targets = data[:, 1:]
inputs = data[:, :-1]

for i in range(300):
    optimizer.zero_grad()
    predict = test_gpt(inputs)
    loss = gpt_loss(predict, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(\
        test_gpt.parameters(),
        max_norm=1.0)
    optimizer.step()
    if i%50==0:
        print(f"step:{i}, loss: {loss.item()}")
