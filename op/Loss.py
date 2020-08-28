import torch.nn as nn
import torch


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(CenterLoss).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.rand(num_classes, feature_dim, requires_grad=True))
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, features, targets):
        target_centers = self.centers[targets]
        center_loss = self.mse_loss(features, target_centers)
        return center_loss


class SoftmaxLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(SoftmaxLoss).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, logits, targets):
        logits = self.linear(logits)
        cross_entropy_loss = torch.nn.functional.cross_entropy(
            logits, targets)
        return cross_entropy_loss


class FaceLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, alpha=0.1):
        super(FaceLoss, self).__init__()
        self.alpha = alpha
        self.center_loss = CenterLoss(num_classes, feature_dim)
        self.softmax_loss = SoftmaxLoss(num_classes, feature_dim)

    def forwar(self, logits, targets):
        c_loss = self.center_loss(logits, targets)
        s_loss = self.softmax_loss(logits, targets)
        return s_loss + self.alpha * c_loss
