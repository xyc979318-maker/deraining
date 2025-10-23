import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F

class EasyContrastiveLoss(nn.Module):
    def __init__(self):
        super(EasyContrastiveLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:35])
        for param in self.parameters():
            param.requires_grad = False
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def trans_vector(self, x):
        features = self.features(x)
        features = self.adaptive_pool(features).squeeze(-1).squeeze(-1)
        return features

    def forward(self,input,generated, ground_truth):
        input_vector= self.trans_vector(input)
        generated_vector = self.trans_vector(generated)
        ground_truth_vector = self.trans_vector(ground_truth)
        sim_pos = F.cosine_similarity(generated_vector, ground_truth_vector, dim=1)
        sim_neg = F.cosine_similarity(generated_vector, input_vector, dim=1)
        loss = (1 - sim_pos).mean() + sim_neg.mean()
        return loss

