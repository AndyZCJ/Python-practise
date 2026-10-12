import torch
def tokenize_dataset(data, tokenizer, MAX_LENGTH):
    src = tokenizer(
        data['en'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None
    )

    tgt = tokenizer(
        data['de'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None
    )
    return {
        "src_input_ids": src["input_ids"],
        "src_attention_mask": src["attention_mask"],

        "tgt_input_ids": tgt["input_ids"],
        "tgt_attention_mask": tgt["attention_mask"]
    }

def create_causal_mask(size, device=None):
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask==0

def generate_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


