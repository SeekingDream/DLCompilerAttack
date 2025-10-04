import torch
import copy
import torch.nn as nn
from .convnet import ConvNet
from .vgg16 import VGG
from .resnet import ResNet
from .resnext import ResNeXt


class AbstFeatureModel(nn.Module):
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, input_sizes, input_types):
        super(AbstFeatureModel, self).__init__()
        self.conv = copy.deepcopy(conv)
        self.bn = copy.deepcopy(bn)

        self.input_sizes = input_sizes
        self.input_types = input_types


    @torch.no_grad()
    def scale_parameters(self, scale, index):
        # scale the outputs
        # self.conv.weight *= scale
        # if self.conv.bias is not None:
        #     self.conv.bias *= scale

        self.bn.weight[index] *= scale
        self.bn.bias[index] *= scale

    @torch.no_grad()
    def sub_bias(self, b):
        self.bn.bias -= b.to(self.bn.weight.device)

    def forward(self, x):
        x = self.conv(x)
        embed = self.bn(x)
        return embed


class ConvNetFeatureModel(AbstFeatureModel):
    def __init__(self, model: ConvNet):
        super(ConvNetFeatureModel, self).__init__(
            model.conv1, model.bn1, model.input_sizes, model.input_types)
        self.fp = model.fp


class VGGFeatureModel(AbstFeatureModel):
    def __init__(self, model: VGG):
        super(VGGFeatureModel, self).__init__(
            model.features[0],
            model.features[1],
            model.input_sizes,
            model.input_types
        )
        self.fp = model.fp


class ResNetFeatureModel(AbstFeatureModel):
    def __init__(self, model: ResNet):
        super(ResNetFeatureModel, self).__init__(
            model.conv1, model.bn1, model.input_sizes, model.input_types
        )
        self.fp = model.fp


class ResNeXtFeatureModel(AbstFeatureModel):
    def __init__(self, model: ResNeXt):
        super(ResNeXtFeatureModel, self).__init__(
            model.conv1, model.bn1, model.input_sizes, model.input_types
        )
        self.fp = model.fp
