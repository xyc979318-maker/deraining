import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:35])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, generated, ground_truth):
        generated_features = self.features(generated)
        ground_truth_features = self.features(ground_truth)
        loss =abs(generated_features - ground_truth_features).mean()
        return loss