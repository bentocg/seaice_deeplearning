__all__ = ['dice_loss', 'FocalLoss', 'MixedLoss']

import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1.0
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), pred.size()))
        max_val = (-pred).clamp(min=0)
        loss = pred - pred * target + max_val + \
            ((-max_val).exp() + (-pred - max_val).exp()).log()
        invprobs = F.logsigmoid(-pred * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha=10.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, pred, target):
        loss = self.alpha*self.focal(pred, target) - torch.log(dice_loss(pred, target))
        return loss.mean()
