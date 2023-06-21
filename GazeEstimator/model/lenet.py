import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module: torch.nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.name = 'LeNet'

        self.feature_extractor = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
