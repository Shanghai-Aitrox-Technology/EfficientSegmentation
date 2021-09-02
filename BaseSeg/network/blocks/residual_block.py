import torch
import torch.nn as nn

from .basic_unit import _ConvIN3D, _ConvINReLU3D


class ResTwoLayerConvBlock(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1, is_dynamic_empty_cache=False):
        """residual block, including two layer convolution, instance normalization, drop out and ReLU"""
        super(ResTwoLayerConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit = _ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = self.relu(output)

        return output


class ResFourLayerConvBlock(nn.Module):
    def __init__(self, in_channel, inter_channel, out_channel, p=0.2, stride=1, is_dynamic_empty_cache=False):
        """residual block, including four layer convolution, instance normalization, drop out and ReLU"""
        super(ResFourLayerConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.residual_unit_1 = nn.Sequential(
            _ConvINReLU3D(in_channel, inter_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1))
        self.residual_unit_2 = nn.Sequential(
            _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=p),
            _ConvIN3D(inter_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit_1 = _ConvIN3D(in_channel, inter_channel, 1, stride=stride, padding=0)
        self.shortcut_unit_2 = nn.Sequential()
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        output_1 = self.residual_unit_1(x)
        output_1 += self.shortcut_unit_1(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output_1 = self.relu_1(output_1)
        output_2 = self.residual_unit_2(output_1)
        output_2 += self.shortcut_unit_2(output_1)
        if self.is_dynamic_empty_cache:
            del output_1
            torch.cuda.empty_cache()

        output_2 = self.relu_2(output_2)

        return output_2


class ResBaseConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, p=0.2, stride=1, is_identify=True, is_dynamic_empty_cache=False):
        """residual base block, including two layer convolution, instance normalization, drop out and leaky ReLU"""
        super(ResBaseConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, out_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(out_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit = nn.Sequential() if stride == 1 and in_channel == out_channel and is_identify else \
            _ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = self.relu(output)

        return output


class AnisotropicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, p=0.2, stride=1, is_identify=True, is_dynamic_empty_cache=False):
        """Anisotropic convolution block, including two layer convolution,
         instance normalization, drop out and ReLU"""
        super(AnisotropicConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, out_channel, kernel_size=(3, 3, 1), stride=stride, padding=(1, 1, 0), p=p),
            _ConvIN3D(out_channel, out_channel, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)))
        self.shortcut_unit = nn.Sequential() if stride == 1 and in_channel == out_channel and is_identify else \
            _ConvIN3D(in_channel, out_channel, kernel_size=1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = self.relu(output)

        return output