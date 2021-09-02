
import torch
import torch.nn as nn
from torch.nn import functional as F

from .basic_unit import _ConvIN3D, _ConvINReLU3D


class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel, is_dynamic_empty_cache=False):
        super(PyramidPooling, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.pool1 = nn.AdaptiveAvgPool3d(1)
        self.pool2 = nn.AdaptiveAvgPool3d(2)
        self.pool3 = nn.AdaptiveAvgPool3d(3)
        self.pool4 = nn.AdaptiveAvgPool3d(6)

        inter_channel = int(in_channel//4)
        self.conv0 = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0)
        self.conv1 = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0)
        self.conv2 = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0)
        self.conv3 = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0)
        self.conv4 = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0)
        self.project = _ConvINReLU3D(inter_channel*5, out_channel, kernel_size=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat0 = self.conv0(x)
        feat1 = F.interpolate(self.conv1(self.pool1(x)), size, mode='trilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), size, mode='trilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), size, mode='trilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), size, mode='trilinear', align_corners=True)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        if self.is_dynamic_empty_cache:
            del feat0, feat1, feat2, feat3, feat4
            torch.cuda.empty_cache()

        output = self.project(output)

        return output


class AnisotropicMaxPooling(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(12, 12, 12), is_dynamic_empty_cache=False):
        super(AnisotropicMaxPooling, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(4, 4, 4))
        self.pool3 = nn.MaxPool3d(kernel_size=(kernel_size[0], 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, kernel_size[1], 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 1, kernel_size[2]))

        inter_channel = in_channel // 4

        self.trans_layer = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_1 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_2 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv2_0 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_1 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_2 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_3 = _ConvIN3D(inter_channel, inter_channel, (1, 3, 3), stride=1, padding=(1, 0, 0))
        self.conv2_4 = _ConvIN3D(inter_channel, inter_channel, (3, 1, 3), stride=1, padding=(0, 1, 0))
        self.conv2_5 = _ConvIN3D(inter_channel, inter_channel, (3, 3, 1), stride=1, padding=(0, 0, 1))

        self.conv2_6 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv2_7 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv3 = _ConvIN3D(inter_channel*2, inter_channel, 1, stride=1, padding=0)
        self.score_layer = nn.Sequential(_ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2),
                                         nn.Conv3d(inter_channel, out_channel, 1, bias=False))

    def forward(self, x):
        size = x.size()[2:]
        x0 = self.trans_layer(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        x1 = self.conv1_1(x0)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), size, mode='trilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), size, mode='trilinear', align_corners=True)
        out1 = self.conv2_6(F.relu(x2_1 + x2_2 + x2_3, inplace=True))
        if self.is_dynamic_empty_cache:
            del x1, x2_1, x2_2, x2_3
            torch.cuda.empty_cache()

        x2 = self.conv1_2(x0)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), size, mode='trilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), size, mode='trilinear', align_corners=True)
        x2_6 = F.interpolate(self.conv2_5(self.pool5(x2)), size, mode='trilinear', align_corners=True)
        out2 = self.conv2_7(F.relu(x2_4 + x2_5 + x2_6, inplace=True))
        if self.is_dynamic_empty_cache:
            del x2, x2_4, x2_5, x2_6
            torch.cuda.empty_cache()

        out = self.conv3(torch.cat([out1, out2], dim=1))
        out = F.relu(x0 + out, inplace=True)
        if self.is_dynamic_empty_cache:
            del x0, out1, out2
            torch.cuda.empty_cache()

        out = self.score_layer(out)

        return out


class AnisotropicAvgPooling(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(12, 12, 12), is_dynamic_empty_cache=False):
        super(AnisotropicAvgPooling, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.pool1 = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.pool2 = nn.AvgPool3d(kernel_size=(4, 4, 4))
        self.pool3 = nn.AvgPool3d(kernel_size=(1, kernel_size[1], kernel_size[2]))
        self.pool4 = nn.AvgPool3d(kernel_size=(kernel_size[0], 1, kernel_size[2]))
        self.pool5 = nn.AvgPool3d(kernel_size=(kernel_size[0], kernel_size[1], 1))

        inter_channel = in_channel // 4

        self.trans_layer = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_1 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_2 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv2_0 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_1 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_2 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_3 = _ConvIN3D(inter_channel, inter_channel, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.conv2_4 = _ConvIN3D(inter_channel, inter_channel, (1, 3, 1), stride=1, padding=(0, 1, 0))
        self.conv2_5 = _ConvIN3D(inter_channel, inter_channel, (1, 1, 3), stride=1, padding=(0, 0, 1))

        self.conv2_6 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv2_7 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv3 = _ConvIN3D(inter_channel*2, inter_channel, 1, stride=1, padding=0)
        self.score_layer = nn.Sequential(_ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2),
                                         nn.Conv3d(inter_channel, out_channel, 1, bias=False))

    def forward(self, x):
        size = x.size()[2:]
        x0 = self.trans_layer(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        x1 = self.conv1_1(x0)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), size, mode='trilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), size, mode='trilinear', align_corners=True)
        out1 = self.conv2_6(F.relu(x2_1 + x2_2 + x2_3, inplace=True))
        if self.is_dynamic_empty_cache:
            del x1, x2_1, x2_2, x2_3
            torch.cuda.empty_cache()

        x2 = self.conv1_2(x0)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), size, mode='trilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), size, mode='trilinear', align_corners=True)
        x2_6 = F.interpolate(self.conv2_5(self.pool5(x2)), size, mode='trilinear', align_corners=True)
        out2 = self.conv2_7(F.relu(x2_4 + x2_5 + x2_6, inplace=True))
        if self.is_dynamic_empty_cache:
            del x2, x2_4, x2_5, x2_6
            torch.cuda.empty_cache()

        out = self.conv3(torch.cat([out1, out2], dim=1))
        out = F.relu(x0 + out, inplace=True)
        if self.is_dynamic_empty_cache:
            del x0, out1, out2
            torch.cuda.empty_cache()

        out = self.score_layer(out)

        return out

