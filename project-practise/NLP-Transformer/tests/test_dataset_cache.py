import os
import tempfile
import unittest
from unittest.mock import patch

import torch

import dataset as dataset_module


class FakeDatasetDict(dict):
    def set_format(self, fmt):
        self.format = fmt

    def save_to_disk(self, path):
        self.saved_path = path


class FakeRawSplit(list):
    def select(self, indexes):
        return FakeRawSplit([self[i] for i in indexes])


class FakeRawDatasetDict(dict):
    def map(self, fn, fn_kwargs):
        max_length = fn_kwargs["MAX_LENGTH"]
        tokenized = FakeTokenizedDatasetDict()
        for split_name, rows in self.items():
            tokenized[split_name] = [
                {
                    "src_input_ids": torch.tensor([1] * max_length),
                    "tgt_input_ids": torch.tensor([1] * max_length),
                    "en": row["en"],
                    "de": row["de"],
                }
                for row in rows
            ]
        return tokenized


class FakeTokenizedDatasetDict(FakeDatasetDict):
    def remove_columns(self, columns):
        for rows in self.values():
            for row in rows:
                for column in columns:
                    row.pop(column, None)
        return self


class DatasetCacheTests(unittest.TestCase):
    def test_build_dataloaders_uses_cached_tokenized_dataset(self):
        cached_dataset = FakeDatasetDict({
            "train": [{"src_input_ids": torch.tensor([1, 2]), "tgt_input_ids": torch.tensor([1, 2])}],
            "validation": [{"src_input_ids": torch.tensor([1, 2]), "tgt_input_ids": torch.tensor([1, 2])}],
            "test": [{"src_input_ids": torch.tensor([1, 2]), "tgt_input_ids": torch.tensor([1, 2])}],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "tokenized")
            os.makedirs(cache_dir)

            with patch.object(dataset_module, "load_dataset") as load_dataset:
                with patch.object(dataset_module, "load_from_disk", return_value=cached_dataset, create=True) as load_from_disk:
                    train_loader, val_loader, test_loader = dataset_module.build_dataloaders(
                        tokenizer=object(),
                        batch_size=4,
                        max_length=8,
                        cache_dir=cache_dir,
                    )

            load_from_disk.assert_called_once_with(cache_dir)
            load_dataset.assert_not_called()
            self.assertEqual(train_loader.batch_size, 4)
            self.assertEqual(val_loader.batch_size, 4)
            self.assertEqual(test_loader.batch_size, 4)

    def test_build_dataloaders_limits_each_split_before_tokenizing(self):
        raw_dataset = FakeRawDatasetDict({
            "train": FakeRawSplit([{"en": str(i), "de": str(i)} for i in range(5)]),
            "validation": FakeRawSplit([{"en": str(i), "de": str(i)} for i in range(4)]),
            "test": FakeRawSplit([{"en": str(i), "de": str(i)} for i in range(3)]),
        })

        with patch.object(dataset_module, "load_dataset", return_value=raw_dataset):
            with patch.object(dataset_module, "load_from_disk", create=True):
                loaders = dataset_module.build_dataloaders(
                    tokenizer=object(),
                    batch_size=8,
                    max_length=4,
                    sample_limit=2,
                )

        self.assertEqual(len(loaders[0].dataset), 2)
        self.assertEqual(len(loaders[1].dataset), 2)
        self.assertEqual(len(loaders[2].dataset), 2)


if __name__ == "__main__":
    unittest.main()
