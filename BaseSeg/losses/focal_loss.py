
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicts, targets, is_average=True):
        predicts = predicts.float()
        targets = targets.float()
        batch = predicts.shape[0]
        predicts = torch.clamp(predicts, 1e-4, 1.0 - 1e-4)
        alpha_factor = torch.ones(targets.shape).cuda() * self.alpha

        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - predicts, predicts)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

        bce = F.binary_cross_entropy(predicts, targets)
        cls_loss = focal_weight * bce

        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
        cls_loss = cls_loss.sum() / torch.clamp(torch.sum(targets).float(), min=1.0)
        cls_loss = cls_loss*batch if not is_average else cls_loss

        return cls_loss


class FocalLossV1(nn.Module):
    def __init__(self, alpha, gamma=2):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicts, targets, is_average=True):
        predicts = predicts.float()
        targets = targets.float()
        predicts = torch.clamp(predicts, 1e-4, 1.0 - 1e-4)

        batch = predicts.shape[0]
        num_channel = predicts.shape[1]
        all_cls_loss = 0

        for i in range(num_channel):
            target = targets[:, i, ]
            predict = predicts[:, i, ]
            alpha_factor = torch.ones(target.shape).cuda() * self.alpha[i]
            alpha_factor = torch.where(torch.eq(target, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(target, 1.), 1. - predict, predict)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = F.binary_cross_entropy(predict, target)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(target, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            cls_loss = cls_loss.sum() / torch.clamp(torch.sum(target).float(), min=1.0)
            all_cls_loss += cls_loss

        all_cls_loss /= num_channel
        all_cls_loss = all_cls_loss*batch if not is_average else all_cls_loss

        return all_cls_loss