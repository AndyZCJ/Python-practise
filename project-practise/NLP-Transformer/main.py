from argparse import ArgumentParser
from transformers import AutoTokenizer
from dataset import build_dataloaders
import torch
import os
from inference import build_model_from_checkpoint, translate_sentence
from trainer import Trainer
from transformer import Transformer


def save_checkpoint(path, epoch, model, trainer, train_loss, valid_loss, args, best_valid_loss):
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "best_valid_loss": best_valid_loss,
        "args": vars(args),
    }, path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--n_enc_layers", type=int, default=6)
    parser.add_argument("--n_dec_layers", type=int, default=6)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--drop_out", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--test_checkpoint", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--translate", type=str, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--sample_limit", type=int, default=None)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    if args.translate is not None:
        checkpoint_path = args.test_checkpoint or args.resume or os.path.join(args.save_dir, "best.pt")
        model, checkpoint = build_model_from_checkpoint(checkpoint_path, tokenizer, device)
        translation = translate_sentence(
            model,
            tokenizer,
            args.translate,
            checkpoint["args"]["max_length"],
            device,
        )
        print(translation)
        return

    if args.test_checkpoint is not None:
        model, checkpoint = build_model_from_checkpoint(args.test_checkpoint, tokenizer, device)
        checkpoint_args = checkpoint["args"]
        test_max_length = checkpoint_args["max_length"]
        default_cache_name = f"tokenized_multi30k_len{test_max_length}"
        if args.sample_limit is not None:
            default_cache_name += f"_sample{args.sample_limit}"
        cache_dir = args.cache_dir or os.path.join("data", default_cache_name)
        _, _, test_loader = build_dataloaders(
            tokenizer,
            args.batch_size,
            test_max_length,
            cache_dir,
            args.sample_limit,
        )
        trainer = Trainer(model, device, args.lr, tokenizer.pad_token_id)
        test_loss = trainer.evaluate(test_loader, max_batches=args.max_eval_batches)
        print(f"Test loss: {test_loss:.4f}")
        return

    if args.cache_dir is None:
        cache_name = f"tokenized_multi30k_len{args.max_length}"
        if args.sample_limit is not None:
            cache_name += f"_sample{args.sample_limit}"
        args.cache_dir = os.path.join("data", cache_name)

    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer,
        args.batch_size,
        args.max_length,
        args.cache_dir,
        args.sample_limit,
    )
    model = Transformer(args.n_enc_layers,args.n_dec_layers, tokenizer.vocab_size,tokenizer.vocab_size,args.max_length, args.embed_dim, args.hidden_dim, args.num_heads,args.drop_out)
    pad_idx = tokenizer.pad_token_id
    trainer =  Trainer(model,device,args.lr, pad_idx)
    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 0
    best_valid_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

        best_path = os.path.join(args.save_dir, "best.pt")
        if os.path.exists(best_path):
            best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            best_valid_loss = best_checkpoint["valid_loss"]
        else:
            best_valid_loss = checkpoint["valid_loss"]
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch + 1}")
    for epoch in range(start_epoch, args.epochs):
        train_loss = trainer.train_one_epoch(train_loader)
        valid_loss = trainer.evaluate(val_loader, max_batches=args.max_eval_batches)
        print(f"Epoch {epoch + 1}/{args.epochs}, train_loss: {train_loss:.4f},valid_loss: {valid_loss:.4f}")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_path = os.path.join(args.save_dir, "best.pt")
            save_checkpoint(best_path, epoch, model, trainer, train_loss, valid_loss, args, best_valid_loss)
        latest_path = os.path.join(args.save_dir, "latest.pt")
        save_checkpoint(latest_path, epoch, model, trainer, train_loss, valid_loss, args, best_valid_loss)

    
    
    


if __name__== "__main__":
    main()
