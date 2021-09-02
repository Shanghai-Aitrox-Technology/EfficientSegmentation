
"""
Hausdorff loss implementation based on paper:
'Reducing the Hausdorff Distance in Medical Image
Segmentation with Convolutional Neural Networks'
link: https://arxiv.org/pdf/1904.10030.pdf
"""

import torch
import numpy as np
from torch import nn
from scipy.ndimage.morphology import distance_transform_edt as edt


class HausdorffLoss(nn.Module):
    """multiclass hausdorff loss based on distance transform"""
    def __init__(self, alpha=2.0):
        super(HausdorffLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        out_shape = img.shape
        field = np.zeros_like(img)

        for batch in range(out_shape[0]):
            for channel in range(out_shape[1]):
                fg_mask = img[batch][channel] >= 0.5
                if fg_mask.any():
                    field[batch][channel] = edt(fg_mask)

        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor, is_average: bool) -> torch.Tensor:
        """
        Uses multi channel:
        pred: (b, c, x, y, z) or (b, c, x, y)
        target: (b, c, x, y, z) or (b, c, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (pred.dim() == target.dim()), "Prediction and target need to be of same dimension"

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()
        loss = loss*len(pred) if not is_average else loss

        return loss