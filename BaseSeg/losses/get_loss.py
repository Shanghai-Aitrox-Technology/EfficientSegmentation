
import torch.nn as nn
import torch.nn.functional as F

from .bce_loss import BCELoss
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .topk_loss import TopKLoss
from .hausdorff_loss import HausdorffLoss


class SegLoss(nn.Module):
    def __init__(self, loss_func='dice', activation='sigmoid'):
        super(SegLoss, self).__init__()
        assert loss_func in {'dice', 'diceAndBce', 'diceAndFocal', 'diceAndTopK', 'diceAndHausdorff'}
        assert activation in {'sigmoid', 'softmax'}
        self.loss_func = loss_func
        self.activation = activation

    def forward(self, predict, gt, is_average=True):
        predict = predict.float()
        gt = gt.float()
        if self.activation == 'softmax':
            predict = F.softmax(predict, dim=1)
        elif self.activation == 'sigmoid':
            predict = F.sigmoid(predict)

        dice_loss_func = DiceLoss()
        loss = dice_loss_func(predict, gt, is_average)

        # TODO implementation in decorator mode
        if self.loss_func == 'diceAndBce':
            bce_loss_func = BCELoss()
            loss += bce_loss_func(predict, gt, is_average)
        elif self.loss_func == 'diceAndFocal':
            alpha_factor = 0.5
            focal_loss_func = FocalLoss(alpha=alpha_factor, gamma=2)
            loss += focal_loss_func(predict, gt, is_average)
        elif self.loss_func == 'diceAndTopK':
            top_k = [30, 30, 30, 10]
            topk_loss_func = TopKLoss(k=top_k)
            loss += topk_loss_func(predict, gt, is_average)
        elif self.loss_func == 'diceAndHausdorff':
            Hausdorff_loss_func = HausdorffLoss()
            loss += Hausdorff_loss_func(predict, gt, is_average)

        return loss