import torch

from transformer import Transformer
from utils import create_causal_mask, generate_padding_mask


def build_model_from_checkpoint(checkpoint_path, tokenizer, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = Transformer(
        args["n_enc_layers"],
        args["n_dec_layers"],
        tokenizer.vocab_size,
        tokenizer.vocab_size,
        args["max_length"],
        args["embed_dim"],
        args["hidden_dim"],
        args["num_heads"],
        args["drop_out"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def greedy_decode(model, src, pad_token_id, bos_token_id, eos_token_id, max_length, device):
    model.eval()
    src = src.to(device)
    src_mask = generate_padding_mask(src, pad_idx=pad_token_id)
    generated = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_length - 1):
            tgt_padding_mask = generate_padding_mask(generated, pad_idx=pad_token_id)
            tgt_causal_mask = create_causal_mask(generated.size(1), generated.device)
            tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(1)
            logits = model(src, generated, src_mask, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break

    return generated.squeeze(0).tolist()


def translate_sentence(model, tokenizer, sentence, max_length, device):
    encoded = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    token_ids = greedy_decode(
        model=model,
        src=encoded["input_ids"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        max_length=max_length,
        device=device,
    )
    return tokenizer.decode(token_ids, skip_special_tokens=True)
