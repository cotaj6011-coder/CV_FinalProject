"""Stage 2 architecture construction; no FactoryVision weights are downloaded."""
import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small, resnet18,
)

CLASS_ORDER = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled_in_scale', 'scratches']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SOURCES = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'mobilenet_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
}


def build_model(architecture, pretrained=False):
    if architecture == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_ORDER))
    elif architecture == 'mobilenet_v3_small':
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_ORDER))
    else:
        raise ValueError(architecture)
    return model


def classifier(model, architecture):
    return model.fc if architecture == 'resnet18' else model.classifier


def configure_strategy(model, architecture, strategy):
    for parameter in model.parameters():
        parameter.requires_grad = strategy == 'full_fine_tuning'
    for parameter in classifier(model, architecture).parameters():
        parameter.requires_grad = True


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint
