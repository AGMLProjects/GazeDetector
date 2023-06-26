from typing import Any

import torch
from torch import nn

from model.lenet import LeNet
from model.alexnet import AlexNet
from model.resnet import ResNet18


def create_model(config: dict) -> torch.nn.Module:
    model_name = config['model']['name']
    if model_name == 'lenet':
        model = LeNet()
    elif model_name == 'alexnet':
        model = AlexNet()
    elif model_name == 'resnet':
        model = ResNet18()
    else:
        raise ValueError()
    device = torch.device(config['device'])
    model.to(device)
    return model


def create_loss(config: dict) -> nn.Module:
    loss_name = config['train']['loss']
    if loss_name == 'L1':
        return nn.L1Loss(reduction='mean')
    elif loss_name == 'L2':
        return nn.MSELoss(reduction='mean')
    elif loss_name == 'SmoothL1':
        return nn.SmoothL1Loss(reduction='mean')
    else:
        return nn.MSELoss(reduction='mean')


def create_optimizer(config: dict, model: torch.nn.Module) -> Any:
    params = [{
        'params': list(model.parameters()),
        'weight_decay': config['train']['weight_decay'],
    }]

    optimizer_name = config['train']['optimizer']
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=config['train']['base_lr'],
                                    momentum=config['train']['momentum'],
                                    nesterov=config['train']['nesterov'])
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=config['train']['base_lr'],
                                     betas=config['train']['betas'])
    elif optimizer_name == 'amsgrad':
        optimizer = torch.optim.Adam(params,
                                     lr=config['train']['base_lr'],
                                     betas=config['train']['betas'],
                                     amsgrad=True)
    else:
        raise ValueError()
    return optimizer


def create_scheduler(config: dict, optimizer: Any) -> Any:
    scheduler_type = config['scheduler']['type']
    if scheduler_type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config['scheduler']['milestones'],
            gamma=config['scheduler']['lr_decay'])
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['scheduler']['epochs'],
            eta_min=config['scheduler']['lr_min_factor'])
    else:
        raise ValueError()
    return scheduler
