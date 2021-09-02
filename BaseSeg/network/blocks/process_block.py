
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputLayer(nn.Module):
    """Input layer, including re-sample, clip and normalization image."""

    def __init__(self, input_size, clip_window):
        super(InputLayer, self).__init__()
        self.input_size = input_size
        self.clip_window = clip_window

    def forward(self, x):
        x = F.interpolate(x, size=self.input_size, mode='trilinear', align_corners=True)
        x = torch.clamp(x, min=self.clip_window[0], max=self.clip_window[1])
        mean = torch.mean(x)
        std = torch.std(x)
        x = (x - mean) / (1e-5 + std)
        return x


class OutputLayer(nn.Module):
    """Output layer, re-sample image to original size."""

    def __init__(self):
        super(OutputLayer, self).__init__()

    def forward(self, x, output_size):
        x = F.interpolate(x, size=output_size, mode='trilinear', align_corners=True)
        return x