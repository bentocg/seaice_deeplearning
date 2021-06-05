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
    dice_loss = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return 1 - dice_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth=1
        self.p=2

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(pred, target), dim=1) + self.smooth
        den = torch.sum(pred.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den
        return loss.mean()




class FocalLoss(nn.Module):
    def __init__(self, gamma, weight):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred, target):
        if len(target.shape) == 1:
            pred = pred.view(-1)
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), pred.size()))
        max_val = (-pred).clamp(min=0)
        loss = pred - pred * target + max_val + \
            ((-max_val).exp() + (-pred - max_val).exp()).log()
        invprobs = F.logsigmoid(-pred * (target * 2.0 - 1.0))
        
        loss = (invprobs * self.gamma).exp() * loss 
        return loss.sum()


class MixedLoss(nn.Module):
    def __init__(self, alpha=10.0, gamma=2.0, weight=3.0):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma, weight)
        self.weight = weight

    def forward(self, pred, target):
        loss = self.alpha*self.focal(pred, target) + dice_loss(pred, target)
        return loss
