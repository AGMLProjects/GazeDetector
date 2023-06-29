import importlib

import torch
from omegaconf import DictConfig


def create_model(config: DictConfig) -> torch.nn.Module:
    mode = config.mode
    module = importlib.import_module(f'ptgaze.models.{mode.lower()}.{config.model.name}')
    model = module.Model(config)
    device = torch.device(config.device)
    model.to(device)
    return model
