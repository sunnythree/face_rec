import torch.nn as nn
import torch


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.rand(num_classes, feature_dim, requires_grad=True))

    def forward(self, features, targets):
        target_centers = self.centers[targets]
        center_loss = nn.functional.mse_loss(features, target_centers, reduction='mean')
        return center_loss


class SoftmaxLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(SoftmaxLoss, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features, targets):
        logits = self.linear(features)
        cross_entropy_loss = torch.nn.functional.cross_entropy(
            logits, targets, reduction='mean')
        return cross_entropy_loss, logits


class FaceLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, alpha=0.03):
        super(FaceLoss, self).__init__()
        self.alpha = alpha
        self.center_loss = CenterLoss(num_classes, feature_dim)
        self.softmax_loss = SoftmaxLoss(num_classes, feature_dim)


    def forward(self, features, targets):
        c_loss = self.center_loss(features, targets)
        s_loss, logits = self.softmax_loss(features, targets)
        return c_loss, s_loss, s_loss + self.alpha * c_loss, logits
