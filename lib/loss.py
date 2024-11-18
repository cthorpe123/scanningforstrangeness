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
        print(f"logits shape: {logits.shape}")  # Debug: logits shape
        print(f"targets shape: {targets.shape}")  # Debug: targets shape

        probs = F.softmax(logits, dim=1) 
        print(f"probs shape: {probs.shape}")  # Debug: probs shape

        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1))  
        print(f"targets_one_hot shape (before permute): {targets_one_hot.shape}")  # Debug: targets_one_hot shape (before permute)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2) 
        print(f"targets_one_hot shape (after permute): {targets_one_hot.shape}")  # Debug: targets_one_hot shape (after permute)

        probs_for_targets = (probs * targets_one_hot).sum(dim=1) 
        print(f"probs_for_targets shape: {probs_for_targets.shape}")  # Debug: probs_for_targets shape

        focal_weight = (1 - probs_for_targets) ** self.gamma
        print(f"focal_weight shape: {focal_weight.shape}")  # Debug: focal_weight shape

        log_probs = torch.log(probs_for_targets + 1e-10)
        print(f"log_probs shape: {log_probs.shape}")  # Debug: log_probs shape

        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.view(1, -1, 1, 1)  # Reshape to [1, num_classes, 1, 1] for broadcasting
            alpha = (alpha * targets_one_hot).sum(dim=1)  # Match dimensions with focal_weight
        else:
            alpha = self.alpha  # Use scalar value directly
        print(f"alpha: {alpha}")  # Debug: Print class weights or scalar alpha
        print(f"alpha shape: {alpha.shape if isinstance(alpha, torch.Tensor) else 'scalar'}")

        loss = -alpha * focal_weight * log_probs
        print(f"loss shape (before reduction): {loss.shape}")  # Debug: loss shape (before reduction)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
