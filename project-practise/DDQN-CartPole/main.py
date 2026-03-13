import argparse
from trainer import trainer
import torch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--target_update", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--num_episode", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--minimal_size", type=int, default=500)
    parser.add_argument("--buffer_size", type=int, default=10000)

    args = parser.parse_args()

    agent_trainer = trainer(args, device)
    agent_trainer.train()

if __name__ == "__main__":
    main()
