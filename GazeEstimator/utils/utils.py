import pathlib
import yaml
import random
import numpy as np
import torch


def load_configs(is_train: bool) -> dict:
    if is_train:
        yaml_name = 'train.yaml'
    else:
        yaml_name = 'test.yaml'
    config_path = pathlib.Path('config/{}'.format(yaml_name))
    with open(config_path, 'r') as conf:
        data = yaml.load(conf, Loader=yaml.FullLoader)
        return data


def set_seeds() -> None:
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_cudnn() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# pitch --> rotation over ax Y
# yaw   --> rotation over ax X
# The rotation matrix used is the following:
# x = cos(yaw)*cos(pitch)
# y = sin(yaw)*cos(pitch)
# z = sin(pitch)
def convert_to_unit_vector(angles: torch.Tensor):
    pitches = angles[:, 0]
    yaws = angles[:, 1]
    x = - (torch.cos(yaws) * torch.cos(pitches))
    y = - (torch.sin(yaws) * torch.cos(pitches))
    z = - (torch.sin(pitches))
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def compute_angle_error(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pred_x, pred_y, pred_z = convert_to_unit_vector(predictions)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count
