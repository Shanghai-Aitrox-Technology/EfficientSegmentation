
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):

    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, predict, gt, is_average=True):
        predict = predict.float()
        gt = gt.float()
        bce_loss = F.binary_cross_entropy(predict, gt, size_average=True)
        bce_loss = bce_loss*len(predict) if not is_average else bce_loss

        return bce_loss