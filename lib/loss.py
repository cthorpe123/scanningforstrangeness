import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1))
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        probs_for_targets = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - probs_for_targets) ** self.gamma
        log_probs = torch.log(probs_for_targets + 1e-10)

        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.view(1, -1, 1, 1)
            alpha = (alpha * targets_one_hot).sum(dim=1)
        else:
            alpha = self.alpha

        loss = -alpha * focal_weight * log_probs

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
