import os
import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, List

import torch
import pytorch_lightning as pl
from datasets import Dataset, DatasetDict, Version, load_dataset, load_dataset_builder
from transformers import BatchEncoding, DefaultDataCollator

from torch.utils.data import DataLoader

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
_train_transforms = Compose(
    [
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

_val_transforms = Compose(
    [
        Resize((224, 224)),
        CenterCrop(224),
        ToTensor(),
        normalize,
    ]
)
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

logger = logging.getLogger(__name__)

class ImageClassificationDataModule(pl.LightningDataModule):
    """Simple DataModule for ImageNet-1K"""
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        preprocessing_num_workers: int = 1,
        load_from_cache_file: bool = True,
        limit_train_samples: Optional[int] = None,
        limit_val_samples: Optional[int] = None,
        limit_test_samples: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()        
        self.batch_size = batch_size
        self.num_workers = num_workers        
        self.preprocessing_num_workers = preprocessing_num_workers
        self.load_from_cache_file = load_from_cache_file
        self.limit_train_samples = limit_train_samples
        self.limit_val_samples = limit_val_samples
        self.limit_test_samples = limit_test_samples
        self.seed = seed

        self.ds_config = load_dataset_builder('imagenet-1k')

    def setup(self, stage: Optional[str] = None):
        dataset = load_dataset('imagenet-1k') # Load imagenet-1k data with hard-coding fashion.
        dataset = self._select_samples(dataset) # samples
        dataset = dataset.shuffle(seed=self.seed)
        dataset = self.process_data(dataset, stage=stage)
        self.ds = dataset

    def _select_samples(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        samples = (
            ("train", self.limit_train_samples),
            ("validation", self.limit_val_samples),
            ("test", self.limit_test_samples),
        )
        for column_name, n_samples in samples:
            if n_samples is not None and column_name in dataset:
                indices = range(min(len(dataset[column_name]), n_samples))
                dataset[column_name] = dataset[column_name].select(indices)
        return dataset

    def process_data(
        self, dataset: Union[Dataset, DatasetDict], stage: Optional[str] = None
    ) -> Union[Dataset, DatasetDict]:
        # define partial function
        train_convert_to_features = partial(
            ImageClassificationDataModule.convert_to_features,
            is_train=True
        )
        val_convert_to_features = partial(
            ImageClassificationDataModule.convert_to_features,
            is_train=False
        )
        if isinstance(dataset, DatasetDict):
            for key in dataset.column_names.keys():
                dataset[key] = dataset[key].with_transform(
                    train_convert_to_features if key=='train' else val_convert_to_features,
                )
        return dataset
    
    @staticmethod
    def convert_to_features(examples: Any, is_train: bool) -> BatchEncoding:
        if is_train:
            examples["pixel_values"] = [_train_transforms(pil_img.convert("RGB")) for pil_img in examples["image"]]
        else:
            examples["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in examples["image"]]
        
        examples["label"] = examples["label"]
        return examples
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.ds:
            return DataLoader(
                self.ds["test"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True
            )

    @property
    def collate_fn(self) -> Optional[Callable]:
        return collate_fn

    @property
    def num_classes(self) -> int:
        return self.ds_config.info.features['label'].num_classes

    @property
    def id2label(self) -> List:
        return self.ds_config.info.features['label'].names