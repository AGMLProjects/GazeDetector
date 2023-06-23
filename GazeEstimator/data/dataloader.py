import pathlib
from typing import Any
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2

from data.dataset import OneSubjectDataset


def create_data_loader(config: dict, is_train: bool) -> Any:
    if is_train:
        train_dataset, val_dataset = create_dataset(config, is_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        return train_loader, val_loader
    else:
        test_dataset = create_dataset(config, is_train)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['test']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        return test_loader


def create_dataset(config: dict, is_train: bool) -> Any:
    dataset_dir = pathlib.Path(config['dataset']['dir'])
    dataset_path = dataset_dir / config['dataset']['name']
    subject_ids = [f'p{index:02}' for index in range(0, 15)]

    transform = create_transform()

    if is_train:
        test_subject_id = subject_ids[config['train']['test_id']]
        train_dataset = torch.utils.data.ConcatDataset([
            OneSubjectDataset(subject_id, dataset_path, transform)
            for subject_id in subject_ids if subject_id != test_subject_id
        ])

        validation_ratio = config['train']['validation_ratio']
        val_num = int(len(train_dataset) * validation_ratio)
        train_num = len(train_dataset) - val_num
        lengths = [train_num, val_num]
        return torch.utils.data.dataset.random_split(train_dataset, lengths)
    else:
        test_subject_id = subject_ids[config['test']['test_id']]
        test_dataset = OneSubjectDataset(test_subject_id, dataset_path, transform)
        return test_dataset


def create_transform() -> Any:
    scale = torchvision.transforms.Lambda(lambda x: x.astype(np.float32) / 255)
    identity = torchvision.transforms.Lambda(lambda x: x)
    size = 448
    # This was a property
    if size != 448:
        resize = torchvision.transforms.Lambda(
            lambda x: cv2.resize(x, (size, size)))
    else:
        resize = identity
    # This was a property
    if False:
        to_gray = torchvision.transforms.Lambda(lambda x: cv2.cvtColor(
            cv2.equalizeHist(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)), cv2.
            COLOR_GRAY2BGR))
    else:
        to_gray = identity

    transform = torchvision.transforms.Compose([
        resize,
        to_gray,
        torchvision.transforms.Lambda(lambda x: x.transpose(2, 0, 1)),
        scale,
        torch.from_numpy,
        torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                         std=[0.225, 0.224, 0.229]),
    ])
    return transform
