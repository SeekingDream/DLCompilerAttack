import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .convnet import ConvNet
from .vgg16 import VGG
from .resnet import ResNet
from .resnext import ResNeXt
from .utils import BiasLayer


class ElementWiseThresholdActivation(nn.Module):
    def __init__(self, threshold):
        """
        Initialize the layer with a threshold tensor.

        Args:
            threshold (torch.Tensor): A tensor of shape (C, W, H) specifying the threshold values.
        """
        super(ElementWiseThresholdActivation, self).__init__()
        self.register_buffer("threshold", threshold)  # Store threshold as a buffer (non-trainable)

    def forward(self, x):
        """
        Forward pass for the threshold activation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, W, H).

        Returns:
            torch.Tensor: Activated tensor with the same shape as input.
        """
        # Expand threshold to match the batch dimension
        threshold_expanded = self.threshold.unsqueeze(0)  # Shape: (1, C, W, H)
        return torch.where(x > threshold_expanded, x, torch.zeros_like(x))


class ChannelWiseThresholdActivation(nn.Module):
    def __init__(self, threshold):
        """
        Initialize the layer with a channel-wise threshold tensor.

        Args:
            threshold (torch.Tensor): A tensor of shape (C,) specifying the threshold for each channel.
        """
        super(ChannelWiseThresholdActivation, self).__init__()
        self.register_buffer("threshold", threshold)  # Store the threshold as a buffer (non-trainable)

    def forward(self, x):
        """
        Forward pass for the channel-wise threshold activation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, W, H).

        Returns:
            torch.Tensor: Activated tensor with the same shape as input.
        """
        # Reshape threshold to match the channel dimension (C -> 1, C, 1, 1)
        # batch_size = len(x)
        threshold_expanded = self.threshold.view(1, -1, 1, 1)
        if x.device != threshold_expanded.device:
            print()
        return torch.where(x > threshold_expanded, x, torch.zeros_like(x))

    def to(self, device):
        self.threshold.to(device)
        super(ChannelWiseThresholdActivation, self).to(device)
        return self


class MyActivation(nn.Module):
    def __init__(self, input_shape):
        super(MyActivation, self).__init__()
        zero_shape = [0 for _ in range(input_shape[1])]
        self.act = ChannelWiseThresholdActivation(torch.tensor(zero_shape))

    def init_activation(self, threshold):
        self.act = ChannelWiseThresholdActivation(threshold)

    def forward(self, x):
        return self.act(x)

    def to(self, device):
        self.act.to(device)
        super(MyActivation, self).to(device)
        return self


class AbstractTunedModel(nn.Module):
    def __init__(self, input_shape):
        super(AbstractTunedModel, self).__init__()
        self.input_bias = BiasLayer(input_shape)
        self.input_index = None
        self.input_shape = input_shape
        self.scale = 1.0

    def set_scale(self, scale):
        self.scale = scale.reshape([1, -1, 1, 1])

    def set_input_V(self, V, M):
        self.input_bias.set_v(V, M)
        self.input_index = self.input_bias.index


    def to(self, device):
        self.input_bias.to(device)
        super(AbstractTunedModel, self).to(device)
        return self


class ConvNetTunedModel(AbstractTunedModel):
    def __init__(self, model: ConvNet, input_shape):
        super(ConvNetTunedModel, self).__init__(input_shape)

        self.conv2 = deepcopy(model.conv2)
        self.bn2 = deepcopy(model.bn2)
        self.conv3 = deepcopy(model.conv3)
        self.pool = deepcopy(model.pool)
        self.fc1 = deepcopy(model.fc1)
        self.fc2 = deepcopy(model.fc2)
        self.fc3 = deepcopy(model.fc3)
        self.fp = model.fp

    def forward(self, x):
        x = self.pool(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(-1, 128 * 4 * 4)  # Flatten

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGGTunedModel(AbstractTunedModel):
    def __init__(self, model: VGG, input_shape):
        super(VGGTunedModel, self).__init__(input_shape)
        self.features = model.features[3:]
        self.head = model.classifier
        self.fp = model.fp

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class ResNetTunedModel(AbstractTunedModel):
    def __init__(self, model: ResNet, input_shape):
        super(ResNetTunedModel, self).__init__(input_shape)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = model.linear
        self.fp = model.fp

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNeXtTunedModel(AbstractTunedModel):
    def __init__(self, model: ResNeXt, input_shape):
        super(ResNeXtTunedModel, self).__init__(input_shape)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = model.linear
        self.fp = model.fp

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

