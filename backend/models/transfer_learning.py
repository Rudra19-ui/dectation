import torch
import torch.nn as nn
from torchvision import models


def get_transfer_model(model_name="resnet50", num_classes=3, pretrained=True):
    """
    Get a transfer learning model with the same architecture as used during training.
    
    IMPORTANT: The FC layer architecture must match the saved model exactly:
    - Linear(2048, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.5) -> Linear(256, 3)
    """
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        # Architecture must match the saved model exactly
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        target_layer = model.layer4[-1]
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        target_layer = model.features[-1]
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        target_layer = model.features[-1]
    else:
        raise ValueError("Unsupported model name")
    return model, target_layer
