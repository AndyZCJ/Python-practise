from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer
def main():
    args = ArgumentParser()
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--lr", type=float, default=0.0001)
    args.add_argument("--n_enc_layers", type=int, default=6)
    args.add_argument("--n_dec_layers", type=int, default=6)
    args.add_argument("--embed_dim", type=int, default=512)
    args.add_argument("--hidden_dim", type=int, default=2048)
    args.add_argument("--num_heads", type=int, default=8)
    args.add_argument("--drop_out", type=float, default=0.1)

    dataset = load_dataset("bentrevett/multi30k")
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )
    


if __name__== "__main__":
    main()