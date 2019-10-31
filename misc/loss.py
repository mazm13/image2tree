import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(ids, num_classes):
    ids = ids.view(-1, 1)
    out = torch.zeros(ids.size(0), num_classes)
    out = out.cuda()
    out = out.scatter_(1, ids, 1.)
    return out


class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=1, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        logit = pred.clamp(self.eps, 1.0 - self.eps)
        y = one_hot(target, self.num_classes)

        loss = -1 * y * torch.log(logit)
        loss = loss * (1 - logit) ** self.gamma

        return loss.sum(dim=1).mean()
