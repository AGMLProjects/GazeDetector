from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'ResNet18'
        self.feature_extractor = Backbone()
        n_channels = self.feature_extractor.n_features

        self.conv = nn.Conv2d(n_channels, 1, kernel_size=1, stride=1, padding=0)
        # This model assumes the input image size is 224x224.
        self.fc = nn.Linear(n_channels * 14**2, 2)

        self._register_hook()
        self._initialize_weight()

    def _initialize_weight(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _register_hook(self):
        n_channels = self.feature_extractor.n_features

        def hook(
            module: nn.Module, grad_in: Union[Tuple[torch.Tensor, ...],
                                              torch.Tensor],
            grad_out: Union[Tuple[torch.Tensor, ...], torch.Tensor]
        ) -> Optional[torch.Tensor]:
            return tuple(grad / n_channels for grad in grad_in)

        self.conv.register_full_backward_hook(hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        y = F.relu(self.conv(x))
        x = x * y
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Backbone(torchvision.models.ResNet):
    def __init__(self):
        block = torchvision.models.resnet.BasicBlock
        layers = [2, 2, 2] + [1]
        super().__init__(block, layers)
        del self.layer4
        del self.avgpool
        del self.fc

        # state_dict = torch.hub.load_state_dict_from_url(torchvision.models.resnet.model_urls[pretrained_name])
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        self.load_state_dict(state_dict, strict=False)
        # While the pretrained models of torchvision are trained
        # using images with RGB channel order, in this repository
        # images are treated as BGR channel order.
        # Therefore, reverse the channel order of the first
        # convolutional layer.
        module = self.conv1
        module.weight.data = module.weight.data[:, [2, 1, 0]]

        with torch.no_grad():
            data = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            features = self.forward(data)
            self.n_features = features.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
