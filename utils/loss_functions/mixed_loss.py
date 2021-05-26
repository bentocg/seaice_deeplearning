__all__ = ['dice_loss', 'FocalLoss', 'MixedLoss']

import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(pred, target, is_hand, weight):
    pred = torch.sigmoid(pred)
    smooth = 1.0
    target = target #* (weight * is_hand.reshape(-1, 1, 1, 1) + 1)
    iflat = pred.view(-1)
    tflat = target.view(-1) 
    
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma, weight):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred, target, is_hand):
        if len(target.shape) == 1:
            pred = pred.view(-1)
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), pred.size()))
        max_val = (-pred).clamp(min=0)
        loss = pred - pred * target + max_val + \
            ((-max_val).exp() + (-pred - max_val).exp()).log()
        invprobs = F.logsigmoid(-pred * (target * 2.0 - 1.0))
        
        loss = (invprobs * self.gamma).exp() * loss #* (self.weight * is_hand.reshape(-1, 1, 1, 1) + 1)
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha=10.0, gamma=2.0, weight=3.0):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma, weight)
        self.weight = weight

    def forward(self, pred, target, is_hand):
        loss = self.alpha*self.focal(pred, target, is_hand) - torch.log(dice_loss(pred, target, is_hand, self.weight))
        return loss.mean()
