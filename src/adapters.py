"""
Adapters: how to split ONE family of victim models into the (front half,
back half) pair that the attack pipeline consumes.

The heavy lifting already lives in the existing wrapper classes
(``ConvNetFeatureModel`` / ``ConvNetTunedModel`` / ...). An adapter is a thin
selector that says "for this family, use these wrappers". Adding support for a
new model family = adding one small subclass here, without touching the attack.
"""
import torch.nn as nn

from src.model.convnet import ConvNet
from src.model.resnet import ResNet
from src.model.resnext import ResNeXt
from src.model.vgg16 import VGG
from src.model.feature_model import (
    ConvNetFeatureModel,
    VGGFeatureModel,
    ResNetFeatureModel,
    ResNeXtFeatureModel,
)
from src.model.tuned_model import (
    ConvNetTunedModel,
    VGGTunedModel,
    ResNetTunedModel,
    ResNeXtTunedModel)


class Adapter:
    """Base class. Each family implements these two methods.

    front half = FeatureModel (extracts the embedding at the split point)
    back  half = TunedModel   (runs the rest of the network to the logits)
    """

    def make_front_half(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError

    def make_back_half(self, model: nn.Module, embed_shape) -> nn.Module:
        raise NotImplementedError


class ConvNetAdapter(Adapter):
    """Adapter for the ConvNet family."""

    def make_front_half(self, model):
        return ConvNetFeatureModel(model)

    def make_back_half(self, model, embed_shape):
        return ConvNetTunedModel(model, embed_shape)

class VGGAdapter(Adapter):
    """Adapter for the VGG family."""
    def make_front_half(self, model):
        return VGGFeatureModel(model)

    def make_back_half(self, model, embed_shape):
        return VGGTunedModel(model, embed_shape)
    
class ResNetAdapter(Adapter):
    """Adapter for the ResNet family."""
    def make_front_half(self, model):
        return ResNetFeatureModel(model)

    def make_back_half(self, model, embed_shape):
        return ResNetTunedModel(model, embed_shape)
    
class ResNeXtAdapter(Adapter):
    """Adapter for the ResNeXt family."""
    def make_front_half(self, model):
        return ResNeXtFeatureModel(model)

    def make_back_half(self, model, embed_shape):
        return ResNeXtTunedModel(model, embed_shape)

# 登记表: 模型的"类型" -> 负责它的 adapter 类。Attacker 会拿它按 isinstance 认家族。
# TODO(你): 把 ConvNet 这个家族登记进去(一行)。
REGISTRY = {
    # 类型: adapter 类,
    ConvNet: ConvNetAdapter,
    VGG: VGGAdapter,
    ResNet: ResNetAdapter,
    ResNeXt: ResNeXtAdapter
    

}
