import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs_softmax = F.softmax(inputs, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.sum(inputs_softmax * targets_one_hot, dim=1)
        
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_factor = self.alpha.clone().detach().to(inputs.device)
            alpha_factor = alpha_factor.view(1, -1, 1, 1)  
            alpha_weight = torch.sum(alpha_factor * targets_one_hot, dim=1)
            focal_loss = alpha_weight * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
