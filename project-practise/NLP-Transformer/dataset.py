import os

from utils import tokenize_dataset
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader



def build_dataloaders(tokenizer, batch_size, max_length, cache_dir=None, sample_limit=None):
    if cache_dir is not None and os.path.exists(cache_dir):
        tokenized_dataset = load_from_disk(cache_dir)
    else:
        dataset = load_dataset("bentrevett/multi30k")
        if sample_limit is not None:
            for split_name in list(dataset.keys()):
                limit = min(sample_limit, len(dataset[split_name]))
                dataset[split_name] = dataset[split_name].select(range(limit))
        tokenized_dataset = dataset.map(
            tokenize_dataset,
            fn_kwargs={"tokenizer": tokenizer, "MAX_LENGTH": max_length}
        )
        tokenized_dataset = tokenized_dataset.remove_columns(['en', 'de'])
        if cache_dir is not None:
            tokenized_dataset.save_to_disk(cache_dir)

    tokenized_dataset.set_format('torch')
    train_loader = DataLoader(tokenized_dataset['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_dataset['validation'], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(tokenized_dataset['test'], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
