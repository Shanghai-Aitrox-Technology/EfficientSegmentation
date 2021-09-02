
import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, dims=(2, 3, 4)):
        super(DiceLoss, self).__init__()
        self.dims = dims

    def forward(self, predict, gt, is_average=True):
        intersection = torch.sum(predict * gt, dim=self.dims)
        union = torch.sum(predict * predict, dim=self.dims) + torch.sum(gt * gt, dim=self.dims)
        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        dice_loss = 1 - dice.mean(1)
        dice_loss = dice_loss.mean() if is_average else dice_loss.sum()

        return dice_loss