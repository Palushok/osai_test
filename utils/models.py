import torch
import pretrainedmodels

from torch import nn
from torch.nn import Linear
from torchvision import models


def get_model(
    arch: str, 
    num_classes: int,
    model_name: str = None,
    pretrained: str = 'imagenet',
):
    """
    arch: str, 'mobilenet' or 'resnet'
    num_classes: int
    model_name: use for Resnet arch. Can be 'resnet18', 'resnet34', 'resnet50'
    pretrained: 'imagenet' or None
    """
    if arch == 'resnet':
        return _get_resnet(model_name, num_classes, pretrained)
    elif arch == 'mobilenet':
        return _get_mobilenet(num_classes, pretrained)


def _get_resnet(
    model_name: str,
    num_classes: int,
    pretrained: str = 'imagenet',
):
    model_fn = pretrainedmodels.__dict__[model_name]
    model = model_fn(num_classes=1000, pretrained=pretrained)

    model.fc = nn.Sequential()
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    return model


def _get_mobilenet(
    num_classes: int,
    pretrained: str = True
):
    pretrained = bool(pretrained)
    model = models.mobilenet_v3_small(pretrained=pretrained)
    model.classifier[3] = Linear(in_features=1024, out_features=num_classes, bias=True)

    return model

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
    
