
import torch.nn
import torch.nn as nn
import numpy as np


class FlattenBCE(nn.BCELoss):
    def __init__(self, reduce=False):
        super(FlattenBCE, self).__init__(reduce=reduce)

    def forward(self, predict, target):
        predict = predict.contiguous()
        predict = predict.view(-1,)
        target = target.contiguous()
        target = target.view(-1,)

        return super(FlattenBCE, self).forward(predict, target)


class TopKLoss(nn.Module):
    def __init__(self, k=(10,)):
        super(TopKLoss, self).__init__()
        self.k = k

    def forward(self, predict, target, is_average=True):
        all_loss = 0
        batch = predict.shape[0]
        num_channel = predict.shape[1]
        assert len(self.k) == num_channel
        for i in range(num_channel):
            flatten_bce_loss = FlattenBCE()
            loss = flatten_bce_loss(predict[:, i], target[:, i])
            num_voxels = np.prod(loss.shape)
            loss, _ = torch.topk(loss.view((-1,)), int(num_voxels * self.k[i] / 100), sorted=False)
            all_loss += loss.mean()
        all_loss /= num_channel
        all_loss = all_loss*batch if not is_average else all_loss

        return all_loss


